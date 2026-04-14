"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型
"""

import logging
import os
import pickle

import faiss
import numpy as np
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ

from core.embeddings import EMBED_MODEL_NAME


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型：
    - 小数据集（<1万）: FlatL2（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat（近似搜索）
    - 大数据集（>10万）: IVFPQ（高效近似搜索）
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """根据向量数量自动选择最优索引类型"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logging.info(f"选择索引类型: {self.index_type}，向量数: {num_vectors}")
        return self.index_type

    def train(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.train(vectors)

    def add(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    解决原代码中 4 个全局变量的管理问题。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表
        self.index_version = "1.0"

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表
            embeddings: 向量数组 (numpy, float32)
        """
        self.clear()
        dimension = embeddings.shape[1]
        num_vectors = len(chunks)

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index
        logging.info(f"FAISS 索引构建完成，共 {self.index.ntotal} 个文本块，类型: {auto_index.index_type}")

    def append_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        追加新文档到现有索引（长期知识库模式）
        """
        if not chunks:
            return

        if not self.is_ready:
            self.build_index(chunks, chunk_ids, metadatas, embeddings)
            return

        dimension = embeddings.shape[1]
        if dimension != self.index.dimension:
            raise ValueError(
                f"向量维度不一致，现有索引维度 {self.index.dimension}，新向量维度 {dimension}，请使用重建模式。"
            )

        existing_ids = set(self.id_order)
        valid_rows = []
        for i, (chunk_id, chunk, meta) in enumerate(zip(chunk_ids, chunks, metadatas)):
            if chunk_id in existing_ids:
                continue
            existing_ids.add(chunk_id)
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)
            valid_rows.append(i)

        if not valid_rows:
            logging.info("未检测到可追加的新文本块（chunk_id 全部已存在）")
            return

        append_embeddings = embeddings[valid_rows]
        self.index.add(append_embeddings)
        logging.info(f"FAISS 索引追加完成，新增 {len(valid_rows)} 个文本块，总计 {self.index.ntotal}")

    def save_index(self, save_dir):
        """
        持久化索引与元数据：
        - index.faiss: 向量索引
        - meta.pkl: 文本内容、元数据、映射和模型信息
        """
        if not self.is_ready:
            logging.warning("当前索引为空，跳过保存")
            return False

        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "meta.pkl")
        index_tmp = index_path + ".tmp"
        meta_tmp = meta_path + ".tmp"

        faiss.write_index(self.index.index, index_tmp)
        meta = {
            "index_version": self.index_version,
            "embed_model_name": EMBED_MODEL_NAME,
            "embed_dim": self.index.dimension,
            "index_type": self.index.index_type,
            "nlist": self.index.nlist,
            "m": self.index.m,
            "nprobe": self.index.nprobe,
            "id_order": list(self.id_order),
            "contents_map": dict(self.contents_map),
            "metadatas_map": dict(self.metadatas_map),
        }
        with open(meta_tmp, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(index_tmp, index_path)
        os.replace(meta_tmp, meta_path)
        logging.info(f"索引已保存到: {save_dir}")
        return True

    def load_index(self, save_dir):
        """
        从磁盘加载索引与元数据，并进行模型一致性校验。
        """
        index_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "meta.pkl")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            logging.info("未发现可加载的索引文件")
            return False, "未发现可加载的索引文件"

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        saved_model = meta.get("embed_model_name", "")
        if saved_model and saved_model != EMBED_MODEL_NAME:
            message = f"嵌入模型不一致（当前: {EMBED_MODEL_NAME}，索引: {saved_model}），请重建索引"
            logging.warning(message)
            return False, message

        raw_index = faiss.read_index(index_path)
        saved_dim = int(meta.get("embed_dim", 0) or 0)
        if saved_dim and raw_index.d != saved_dim:
            message = f"索引维度不一致（FAISS: {raw_index.d}，元数据: {saved_dim}），请重建索引"
            logging.warning(message)
            return False, message

        id_order = list(meta.get("id_order", []))
        if raw_index.ntotal != len(id_order):
            message = f"索引与 id_order 数量不一致（index={raw_index.ntotal}, id_order={len(id_order)}）"
            logging.warning(message)
            return False, message

        auto_index = AutoFaissIndex(dimension=raw_index.d)
        auto_index.index = raw_index
        auto_index.index_type = meta.get("index_type") or self._infer_index_type(raw_index)
        auto_index.nlist = meta.get("nlist")
        auto_index.m = meta.get("m")
        auto_index.nprobe = meta.get("nprobe") or self._default_nprobe(auto_index.index_type, auto_index.nlist)

        self.index = auto_index
        self.id_order = id_order
        self.contents_map = dict(meta.get("contents_map", {}))
        self.metadatas_map = dict(meta.get("metadatas_map", {}))

        logging.info(f"索引加载成功，共 {self.index.ntotal} 个文本块，类型: {auto_index.index_type}")
        return True, "索引加载成功"

    @staticmethod
    def _infer_index_type(raw_index):
        name = type(raw_index).__name__
        if "IVFPQ" in name:
            return "IVFPQ"
        if "IVFFlat" in name:
            return "IVFFlat"
        return "FlatL2"

    @staticmethod
    def _default_nprobe(index_type, nlist):
        if index_type == "IVFPQ":
            return min(32, max(1, int((nlist or 1) * 0.05)))
        if index_type == "IVFFlat":
            return min(10, max(1, int((nlist or 1) * 0.1)))
        return 1

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量

        Returns:
            (docs, doc_ids, metadatas)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logging.error(f"FAISS 检索错误: {str(e)}")
            return [], [], []

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        logging.info("向量存储已清空")


# 模块级单例
vector_store = VectorStore()
