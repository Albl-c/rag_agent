"""
重排序器 —— 对检索结果进行二次精排

学习要点：
- 两阶段检索（Recall + Rerank）是工业界常用的范式
- Recall 阶段用高效检索（FAISS/BM25）从大量文档中召回候选
- Rerank 阶段用更精确的模型（交叉编码器/LLM）对候选精排
- 交叉编码器比双塔模型更精确，但速度更慢（适合对少量候选精排）
"""

import logging
import re
import threading
from functools import lru_cache
from config import OLLAMA_MODEL_NAME, RERANK_METHOD, RERANK_MODEL_NAME

# 交叉编码器（懒加载 + 线程安全）
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """懒加载交叉编码器模型（双重检查锁定，线程安全）"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
                    logging.info(f"交叉编码器加载成功: {RERANK_MODEL_NAME}")
                except Exception as e:
                    logging.error(f"加载交叉编码器失败: {str(e)}")
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """使用交叉编码器对检索结果进行重排序"""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("交叉编码器不可用，跳过重排序")
        return _fallback_results(doc_ids, docs, metadata_list)

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {str(e)}")
        return _fallback_results(doc_ids, docs, metadata_list)


@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """使用 LLM 对查询和文档的相关性进行评分（带缓存）"""
    from utils.network import get_session
    try:
        prompt = f"""给定以下查询和文档片段，评估它们的相关性。
        评分标准：0分表示完全不相关，10分表示高度相关。
        只需返回一个0-10之间的整数分数，不要有任何其他解释。

        查询: {query}
        文档片段: {doc}
        相关性分数(0-10):"""

        response = get_session().post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180
        )
        result = response.json().get("response", "").strip()
        try:
            return max(0, min(10, float(result)))
        except ValueError:
            match = re.search(r'\b([0-9]|10)\b', result)
            return float(match.group(1)) if match else 5.0
    except Exception as e:
        logging.error(f"LLM评分失败: {str(e)}")
        return 5.0


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """使用 LLM 逐一评分进行重排序"""
    if not docs:
        return []
    results = []
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        score = get_llm_relevance_score(query, doc)
        results.append((doc_id, {'content': doc, 'metadata': meta, 'score': score / 10.0}))
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    return results[:top_k]


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """对检索结果进行重排序（统一入口）"""
    if method is None:
        method = RERANK_METHOD

    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids, docs, metadata_list):
    """回退方案：按原始顺序返回"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
