"""
轻量级 RAG 评测脚本

指标：
1) Hit@5（基于 gold_sources 与 Top5 检索 source 文件名匹配）
2) MRR（首个命中排名倒数）
3) Faithfulness（答案是否忠实于检索上下文）
4) Relevancy（答案是否切题）
5) AnswerSuccessRate（关键词命中率 >= 阈值，默认 0.4）
6) AvgLatencyMs（query_answer 端到端平均耗时）

用法示例：
python eval/light_eval.py ^
  --data-dir files ^
  --questions eval/questions.jsonl ^
  --chunk-size 600 ^
  --chunk-overlap 80 ^
  --model-choice ollama ^
  --enable-bm25 false ^
  --auto-route true
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import requests
from config import OLLAMA_MODEL_NAME
from core.document_loader import extract_text
from core.text_splitter import split_text
from core.chunk_router import route_chunk_strategy
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.retriever import recursive_retrieval
from core.generator import query_answer, call_llm_simple


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def load_questions(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise ValueError("问题集为空，请检查 questions.jsonl")
    return items


def collect_documents(data_dir: str) -> List[str]:
    all_files = []
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if os.path.isfile(full) and name.lower().endswith((".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls", ".pptx")):
            all_files.append(full)
    if not all_files:
        raise ValueError(f"目录 {data_dir} 未找到可处理文件")
    return all_files


def build_index(
    file_paths: List[str],
    chunk_size: int,
    chunk_overlap: int,
    enable_bm25: bool,
    auto_route: bool
) -> Tuple[int, int]:
    vector_store.clear()
    bm25_manager.clear()

    all_chunks: List[str] = []
    all_ids: List[str] = []
    all_metadatas: List[Dict] = []

    for idx, fp in enumerate(file_paths, 1):
        text = extract_text(fp)
        if not text:
            continue
        selected_strategy = "configured"
        if auto_route:
            selected_strategy, _ = route_chunk_strategy(os.path.basename(fp), text)
            chunks = split_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=selected_strategy
            )
        else:
            chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        source = os.path.basename(fp)
        doc_id = f"eval_doc_{idx}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": source,
            "doc_id": doc_id,
            "chunk_strategy": selected_strategy
        } for _ in chunks]

        all_chunks.extend(chunks)
        all_ids.extend(chunk_ids)
        all_metadatas.extend(metadatas)

    if not all_chunks:
        raise ValueError("所有文档解析后为空，无法建索引")

    embeddings = encode_texts(all_chunks, show_progress=False)
    vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)

    if enable_bm25:
        bm25_manager.build_index(all_chunks, all_ids)

    return len(file_paths), len(all_chunks)


def normalize_list(values: List[str]) -> List[str]:
    return [str(v).strip().lower() for v in values if str(v).strip()]


def _call_judge_llm(prompt: str, model_choice: str) -> str:
    """
    固定 judge 模型 + temperature=0，减少波动。
    优先走 Ollama 原生接口，失败时回退到项目现有 call_llm_simple。
    """
    judge_model = os.getenv("EVAL_JUDGE_MODEL", OLLAMA_MODEL_NAME).strip() or OLLAMA_MODEL_NAME
    host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/chat"
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("message", {}).get("content", "")).strip()
    except Exception:
        # 回退：尽量保证脚本可运行
        return str(call_llm_simple(prompt, model_choice=model_choice))


def _custom_quality_eval(question: str, answer: str, contexts: List[str], model_choice: str) -> Tuple[float, float, str, str]:
    """
    自定义连续评分（0~1）：
    - Faithfulness：答案是否基于检索上下文，是否存在上下文不支持的断言（抗幻觉）
    - Relevancy：答案是否直接回答用户问题（切题程度）
    """
    context_text = "\n\n".join(contexts[:5]) if contexts else "无上下文"
    prompt = f"""你是RAG评测裁判。请按以下标准评分（0~1，允许小数）：

Faithfulness（忠实度）：
- 评估答案是否完全基于“检索上下文”。
- 若出现上下文不支持、夸大、编造、偷换结论，降分。
- 分数越高，说明答案越可靠、幻觉越少。

Relevancy（相关性）：
- 评估答案是否直接回答“用户问题”。
- 只答边缘信息、跑题、避答，降分。
- 分数越高，说明越切题。

请严格输出 JSON（不要任何额外文本）：
{{
  "faithfulness": 0.0,
  "relevancy": 0.0,
  "reason": "一句话说明打分依据"
}}

[Question]
{question}

[Contexts]
{context_text}

[Answer]
{answer}
"""
    text = _call_judge_llm(prompt=prompt, model_choice=model_choice)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return 0.0, 0.0, "custom_judge_llm", "judge_output_not_json"
    try:
        payload = json.loads(match.group(0))
        faith = float(payload.get("faithfulness", 0.0))
        rel = float(payload.get("relevancy", 0.0))
        reason = str(payload.get("reason", "")).strip()
        return (
            max(0.0, min(1.0, faith)),
            max(0.0, min(1.0, rel)),
            "custom_judge_llm",
            reason,
        )
    except Exception:
        return 0.0, 0.0, "custom_judge_llm", "judge_output_parse_failed"


def evaluate(
    questions: List[Dict],
    model_choice: str,
    enable_web_search: bool,
    top_k: int,
    keyword_hit_threshold: float,
) -> Dict:
    hit_count = 0
    mrr_sum = 0.0
    answer_success_count = 0
    faithfulness_scores: List[float] = []
    relevancy_scores: List[float] = []
    latency_ms_list: List[float] = []
    details: List[Dict] = []

    for idx, q in enumerate(questions, 1):
        question = q.get("question", "").strip()
        if not question:
            continue

        gold_sources = normalize_list(q.get("gold_sources", []))
        expected_keywords = normalize_list(q.get("expected_keywords", []))

        # 1) 检索结果用于 Hit@5
        all_contexts, _, all_metadata = recursive_retrieval(
            initial_query=question,
            enable_web_search=enable_web_search,
            model_choice=model_choice,
        )
        retrieved_sources = []
        for meta in all_metadata:
            source = str(meta.get("source", "")).strip().lower()
            if source and source not in retrieved_sources:
                retrieved_sources.append(source)
            if len(retrieved_sources) >= top_k:
                break

        hit = bool(set(gold_sources) & set(retrieved_sources)) if gold_sources else False
        if hit:
            hit_count += 1

        reciprocal_rank = 0.0
        if gold_sources:
            for rank, source in enumerate(retrieved_sources, 1):
                if source in gold_sources:
                    reciprocal_rank = 1.0 / rank
                    break
        mrr_sum += reciprocal_rank

        # 2) 回答 + 时延
        t0 = time.perf_counter()
        answer = query_answer(
            question=question,
            enable_web_search=enable_web_search,
            model_choice=model_choice,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms_list.append(latency_ms)

        faithfulness, relevancy, eval_engine, eval_reason = _custom_quality_eval(
            question=question, answer=str(answer), contexts=all_contexts[:top_k], model_choice=model_choice
        )

        faithfulness_scores.append(faithfulness)
        relevancy_scores.append(relevancy)

        answer_lower = str(answer).lower()
        answer_success = False
        keyword_hit_ratio = 0.0
        matched_keywords: List[str] = []
        if expected_keywords:
            matched_keywords = [kw for kw in expected_keywords if kw in answer_lower]
            keyword_hit_ratio = len(matched_keywords) / len(expected_keywords)
            answer_success = keyword_hit_ratio >= keyword_hit_threshold
        else:
            # 没有标 expected_keywords 时不计入成功，避免虚高
            answer_success = False
        if answer_success:
            answer_success_count += 1

        details.append(
            {
                "id": q.get("id", f"q_{idx}"),
                "question": question,
                "gold_sources": gold_sources,
                "retrieved_top_sources": retrieved_sources,
                "hit_at_k": hit,
                "reciprocal_rank": round(reciprocal_rank, 4),
                "faithfulness_score": round(faithfulness, 4),
                "relevancy_score": round(relevancy, 4),
                "quality_eval_engine": eval_engine,
                "quality_eval_reason": eval_reason,
                "expected_keywords": expected_keywords,
                "matched_keywords": matched_keywords,
                "keyword_hit_ratio": round(keyword_hit_ratio, 4),
                "answer_success": answer_success,
                "latency_ms": round(latency_ms, 2),
            }
        )

    total = len(details)
    avg_latency = sum(latency_ms_list) / len(latency_ms_list) if latency_ms_list else 0.0
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    return {
        "total_questions": total,
        "hit_at_k": round(hit_count / total, 4) if total else 0.0,
        "mrr": round(mrr_sum / total, 4) if total else 0.0,
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_relevancy": round(avg_relevancy, 4),
        "answer_success_rate": round(answer_success_count / total, 4) if total else 0.0,
        "avg_latency_ms": round(avg_latency, 2),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="轻量级 RAG 评测脚本")
    parser.add_argument("--data-dir", required=True, help="待建库文档目录，如 files 或 test_data")
    parser.add_argument("--questions", required=True, help="问题集 jsonl 路径")
    parser.add_argument("--output", default="eval/results/latest.json", help="评测结果输出文件")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--model-choice", default="ollama", choices=["ollama", "siliconflow"])
    parser.add_argument("--enable-web-search", default="false")
    parser.add_argument("--enable-bm25", default="false")
    parser.add_argument("--auto-route", default="false", help="是否启用自动分块策略路由")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--keyword-hit-threshold",
        type=float,
        default=0.4,
        help="答案成功判定阈值：命中关键词数/总关键词数，范围 0~1，默认 0.4",
    )
    args = parser.parse_args()

    enable_web_search = str2bool(args.enable_web_search)
    enable_bm25 = str2bool(args.enable_bm25)
    auto_route = str2bool(args.auto_route)

    file_paths = collect_documents(args.data_dir)
    questions = load_questions(args.questions)
    file_count, chunk_count = build_index(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_bm25=enable_bm25,
        auto_route=auto_route,
    )

    result = evaluate(
        questions=questions,
        model_choice=args.model_choice,
        enable_web_search=enable_web_search,
        top_k=args.top_k,
        keyword_hit_threshold=max(0.0, min(1.0, args.keyword_hit_threshold)),
    )
    result["experiment"] = {
        "data_dir": args.data_dir,
        "file_count": file_count,
        "chunk_count": chunk_count,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "enable_bm25": enable_bm25,
        "auto_route": auto_route,
        "model_choice": args.model_choice,
        "enable_web_search": enable_web_search,
        "top_k": args.top_k,
        "keyword_hit_threshold": max(0.0, min(1.0, args.keyword_hit_threshold)),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Eval Summary ===")
    print(f"Questions: {result['total_questions']}")
    print(f"Hit@{args.top_k}: {result['hit_at_k']}")
    print(f"MRR: {result['mrr']}")
    print(f"Faithfulness: {result['avg_faithfulness']}")
    print(f"Relevancy: {result['avg_relevancy']}")
    print(f"AnswerSuccessRate: {result['answer_success_rate']}")
    print(f"AvgLatencyMs: {result['avg_latency_ms']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
