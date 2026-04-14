"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 密钥配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# Ollama 格式: deepseek-r1:8b | SiliconFlow 格式: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
SILICONFLOW_MODEL_NAME = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
# RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")
RERANK_METHOD = os.getenv("RERANK_METHOD", "none")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 425          # 文本分块大小（字符数，评测最优）
CHUNK_OVERLAP = 45        # 相邻分块的重叠字符数（评测最优）
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")  # 可选: recursive | structured_recursive | markdown_header | llama_nodes
HYBRID_ALPHA = 0.7        # 混合检索中语义检索的权重（0-1）

RETRIEVAL_TOP_K = 10      # 检索返回的候选文档数量
RERANK_TOP_K = 5          # 重排序后保留的文档数量

MAX_RETRIEVAL_ITERATIONS = 1  # 递归检索的最大迭代轮数（单轮，稳定配置）
# MAX_RETRIEVAL_ITERATIONS = 3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    自动检测可用的 LLM 后端，返回默认模型选择

    检测优先级：
    1. SiliconFlow API Key 已配置 → 默认使用云端 API
    2. 本地 Ollama 服务可用 → 默认使用本地模型
    3. 都不可用 → 返回 siliconflow 并提示用户配置
    """
    if SILICONFLOW_API_KEY and SILICONFLOW_API_KEY.strip() and not SILICONFLOW_API_KEY.startswith("Your"):
        logging.info("✅ 检测到 SiliconFlow API Key，默认使用云端模型")
        return "siliconflow"

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logging.info("✅ 检测到本地 Ollama 服务，默认使用本地模型")
            return "ollama"
    except Exception:
        pass

    logging.warning("⚠️ 未检测到可用 LLM 后端，请配置 SiliconFlow API Key 或启动 Ollama")
    return "siliconflow"

DEFAULT_MODEL_CHOICE = detect_default_model()
