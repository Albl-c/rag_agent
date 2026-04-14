"""
LLM 调用 —— 大模型回答生成（Ollama + SiliconFlow）

学习要点：
- Prompt Engineering：如何构建高质量的提示词模板
- 流式输出 vs 非流式输出的区别
- 多模型适配：本地 Ollama 和云端 SiliconFlow API 的对接
"""

import json
import logging
import requests
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL,
    SILICONFLOW_MODEL_NAME, OLLAMA_MODEL_NAME
)
from utils.network import get_session
from core.retriever import recursive_retrieval
from core.vector_store import vector_store
from features.conflict_detector import detect_conflicts, evaluate_source_credibility
from features.thinking_chain import process_thinking_content


def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """调用 SiliconFlow 云端 API 获取回答"""
    if not SILICONFLOW_API_KEY:
        logging.error("未设置 SILICONFLOW_API_KEY")
        return "错误：未配置 SiliconFlow API 密钥。"

    try:
        payload = {
            "model": SILICONFLOW_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.7, "top_k": 50,
            "frequency_penalty": 0.5, "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY.strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(SILICONFLOW_API_URL, data=json_payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                return f"{content}<think>{reasoning}</think>"
            return content
        return "API返回结果格式异常"

    except requests.exceptions.RequestException as e:
        logging.error(f"调用SiliconFlow API时出错: {str(e)}")
        return f"调用API时出错: {str(e)}"
    except Exception as e:
        logging.error(f"SiliconFlow API 未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"


def call_llm_simple(prompt, model_choice="siliconflow"):
    """简单的 LLM 调用（用于递归检索中的查询改写判断）"""
    if model_choice == "siliconflow":
        result = call_siliconflow_api(prompt)
        result = result.strip() if isinstance(result, str) else result[0].strip()
        if "<think>" in result:
            result = result.split("<think>")[0].strip()
        return result
    else:
        response = get_session().post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180
        )
        return response.json().get("response", "").strip()


def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected):
    """构建提示词"""
    prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""

    return prompt_template.format(
        context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else (
            "网络搜索结果" if enable_web_search else "本地文档"),
        context=context if context else (
            "网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
        question=question,
        time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
        conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
    )


def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """构建上下文和来源信息"""
    context_parts = []
    sources_for_conflict = []

    for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
        source_type = metadata.get('source', '本地文档')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            url = metadata.get('url', '未知URL')
            title = metadata.get('title', '未知标题')
            context_parts.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
        else:
            source = metadata.get('source', '未知来源')
            context_parts.append(f"[本地文档: {source}]\n{doc}")
            source_item['source'] = source

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict


def query_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None):
    """
    问答处理主流程（非流式）

    完整流程：递归检索 → 构建上下文 → 矛盾检测 → 构建Prompt → LLM生成
    """
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ 知识库为空，请先上传文档。"

        if progress:
            progress(0.3, desc="执行递归检索...")

        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected)

        if progress:
            progress(0.8, desc="生成回答...")

        if model_choice == "siliconflow":
            result = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
        else:
            response = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=180, headers={'Connection': 'close'}
            )
            response.raise_for_status()
            result = str(response.json().get("response", "未获取到有效回答"))

        return process_thinking_content(result)

    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except Exception as e:
        return f"系统错误: {str(e)}"


def stream_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None):
    """问答处理主流程（流式，用于 Gradio generator 模式）"""
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
            return

        if progress:
            progress(0.3, desc="执行递归检索...")

        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected)

        if model_choice == "siliconflow":
            full_answer = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            yield process_thinking_content(full_answer), "完成!"
        else:
            response = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": True},
                timeout=120, stream=True
            )
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode()).get("response", "")
                    full_answer += chunk
                    if "<think>" in full_answer and "</think>" in full_answer:
                        yield process_thinking_content(full_answer), "生成回答中..."
                    else:
                        yield full_answer, "生成回答中..."

            yield process_thinking_content(full_answer), "完成!"

    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"
