"""
向量化模型 —— 将文本映射到高维向量空间

学习要点：
- Embedding 将文本转换为固定维度的向量，使语义相似的文本在向量空间中距离更近
- intfloat/multilingual-e5-base 是多语言模型（768维），适合中英文检索
- 首次运行时模型会自动下载（约 80MB），需要网络连接
"""

import logging
import numpy as np
from functools import lru_cache

# 当前项目固定使用该 embedding 模型，不再切换
# 使用本地目录避免首次运行时的联网下载重试。
EMBED_MODEL_NAME = 'models/multilingual-e5-base'


@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    from sentence_transformers import SentenceTransformer
    logging.info(f"加载向量化模型: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info(f"向量化模型加载完成，输出维度: {model.get_sentence_embedding_dimension()}")
    return model


def _format_e5_documents(texts):
    """E5 系列建议为文档加 passage 前缀。"""
    if EMBED_MODEL_NAME.startswith("intfloat/multilingual-e5"):
        return [f"passage: {text}" for text in texts]
    return texts


def _format_e5_query(query):
    """E5 系列建议为查询加 query 前缀。"""
    if EMBED_MODEL_NAME.startswith("intfloat/multilingual-e5"):
        return f"query: {query}"
    return query


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    model_inputs = _format_e5_documents(texts)
    embeddings = model.encode(model_inputs, show_progress_bar=show_progress)
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    将单个查询文本编码为向量

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    model = get_embed_model()
    model_input = _format_e5_query(query)
    embedding = model.encode([model_input])
    return np.array(embedding).astype('float32')
