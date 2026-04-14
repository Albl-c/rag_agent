"""
自动路由检索侧快速评测（不调用生成模型）。

核心指标：
1) Hit@K
2) MRR
3) AvgRetrievalMs
4) ChunkCount

用法：
python -m eval.route_retrieval_eval --data-dir test_data --questions eval/questions.test_data20.jsonl --auto-route true --output eval/results/route_retrieval.json
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from core.document_loader import extract_text
from core.text_splitter import split_text
from core.chunk_router import route_chunk_strategy
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.retriever import recursive_retrieval
from core.bm25_index import bm25_manager


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
    files = []
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if os.path.isfile(full) and name.lower().endswith((".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls", ".pptx")):
            files.append(full)
    if not files:
        raise ValueError(f"目录 {data_dir} 未找到可处理文件")
    return files


def normalize_list(values: List[str]) -> List[str]:
    return [str(v).strip().lower() for v in values if str(v).strip()]


def build_index(
    file_paths: List[str],
    chunk_size: int,
    chunk_overlap: int,
    auto_route: bool,
    enable_bm25: bool,
) -> Tuple[int, int, Dict[str, int]]:
    vector_store.clear()
    bm25_manager.clear()

    all_chunks: List[str] = []
    all_ids: List[str] = []
    all_metadatas: List[Dict] = []
    route_counter: Dict[str, int] = {"recursive": 0, "structured_recursive": 0, "markdown_header": 0, "llama_nodes": 0}

    for idx, fp in enumerate(file_paths, 1):
        text = extract_text(fp)
        if not text:
            continue

        source = os.path.basename(fp)
        strategy = "configured"
        if auto_route:
            strategy, _ = route_chunk_strategy(source, text)
            route_counter[strategy] = route_counter.get(strategy, 0) + 1
            chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
        else:
            route_counter["recursive"] += 1
            chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        doc_id = f"eval_doc_{idx}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "doc_id": doc_id, "chunk_strategy": strategy} for _ in chunks]

        all_chunks.extend(chunks)
        all_ids.extend(chunk_ids)
        all_metadatas.extend(metadatas)

    if not all_chunks:
        raise ValueError("所有文档解析后为空，无法建索引")

    embeddings = encode_texts(all_chunks, show_progress=False)
    vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)
    if enable_bm25:
        bm25_manager.build_index(all_chunks, all_ids)

    return len(file_paths), len(all_chunks), route_counter


def evaluate_retrieval(
    questions: List[Dict],
    top_k: int,
    enable_bm25: bool,
) -> Dict:
    hit_count = 0
    mrr_sum = 0.0
    retrieval_ms: List[float] = []

    for q in questions:
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        gold_sources = normalize_list(q.get("gold_sources", []))

        t0 = time.perf_counter()
        _, _, all_metadata = recursive_retrieval(
            initial_query=question,
            enable_web_search=False,
            model_choice="ollama",
        )
        dt = (time.perf_counter() - t0) * 1000.0
        retrieval_ms.append(dt)

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

        rr = 0.0
        if gold_sources:
            for rank, source in enumerate(retrieved_sources, 1):
                if source in gold_sources:
                    rr = 1.0 / rank
                    break
        mrr_sum += rr

    total = len(questions)
    avg_ms = sum(retrieval_ms) / len(retrieval_ms) if retrieval_ms else 0.0
    return {
        "total_questions": total,
        "hit_at_k": round(hit_count / total, 4) if total else 0.0,
        "mrr": round(mrr_sum / total, 4) if total else 0.0,
        "avg_retrieval_ms": round(avg_ms, 2),
        "enable_bm25": enable_bm25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="自动路由检索侧快速评测")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=425)
    parser.add_argument("--chunk-overlap", type=int, default=45)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--auto-route", default="false")
    parser.add_argument("--enable-bm25", default="false")
    args = parser.parse_args()

    auto_route = str2bool(args.auto_route)
    enable_bm25 = str2bool(args.enable_bm25)

    file_paths = collect_documents(args.data_dir)
    questions = load_questions(args.questions)
    file_count, chunk_count, route_counter = build_index(
        file_paths=file_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        auto_route=auto_route,
        enable_bm25=enable_bm25,
    )
    metrics = evaluate_retrieval(
        questions=questions,
        top_k=args.top_k,
        enable_bm25=enable_bm25,
    )
    result = {
        **metrics,
        "experiment": {
            "data_dir": args.data_dir,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "auto_route": auto_route,
            "enable_bm25": enable_bm25,
            "route_counter": route_counter,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Retrieval Eval Summary ===")
    print(f"Questions: {result['total_questions']}")
    print(f"Hit@{args.top_k}: {result['hit_at_k']}")
    print(f"MRR: {result['mrr']}")
    print(f"AvgRetrievalMs: {result['avg_retrieval_ms']}")
    print(f"ChunkCount: {result['experiment']['chunk_count']}")
    print(f"AutoRoute: {result['experiment']['auto_route']}")
    print(f"RouteCounter: {result['experiment']['route_counter']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
