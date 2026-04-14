"""
联网搜索 —— 通过 SerpAPI 获取实时网络信息

学习要点：
- RAG 的"R"不限于本地文档，也可以从网络获取实时信息
- SerpAPI 是 Google 搜索的 API 封装，需要注册获取 API Key
- 网络搜索结果不进入 FAISS 索引，仅作为文本上下文提供给 LLM
"""

import logging
import requests
from config import SERPAPI_KEY, SEARCH_ENGINE


def check_serpapi_key():
    """检查是否配置了有效的 SERPAPI_KEY"""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != "" and not SERPAPI_KEY.startswith("Your")


def serpapi_search(query, num_results=5):
    """执行 SerpAPI 搜索"""
    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量")
    try:
        params = {
            "engine": SEARCH_ENGINE, "q": query, "api_key": SERPAPI_KEY,
            "num": num_results, "hl": "zh-CN", "gl": "cn"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as e:
        logging.error(f"网络搜索失败: {str(e)}")
        return []


def _parse_serpapi_results(data):
    """解析 SerpAPI 返回的原始数据"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"), "url": item.get("link"),
                "snippet": item.get("snippet"), "timestamp": item.get("date")
            })
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"), "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"), "source": "knowledge_graph"
        })
    return results


def search_web(query, num_results=5):
    """执行网络搜索（结果不加入 FAISS 索引，仅作为上下文）"""
    results = serpapi_search(query, num_results)
    if not results:
        logging.info("网络搜索没有返回结果")
    else:
        logging.info(f"网络搜索返回 {len(results)} 条结果")
    return results
