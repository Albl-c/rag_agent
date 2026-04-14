"""
文本分块器 —— 将长文本切分为检索友好的片段

学习要点：
- chunk_size：每个片段的最大字符数。过大则检索粒度粗，过小则上下文缺失
- chunk_overlap：相邻片段的重叠字符数。避免关键信息被切断
- separators：按优先级尝试的分割符。中文文档应包含中文标点
- chunk_strategy：通过环境变量切换不同切分策略，默认 recursive
"""

import logging
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY


def _split_with_recursive(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return splitter.split_text(text)


def _split_with_structured_recursive(text, chunk_size, chunk_overlap):
    """
    语料结构感知切分（适配医学摘要常见标签）：
    1) 先按 OBJECTIVE/METHODS/RESULTS/CONCLUSIONS 等结构标签分段
    2) 再对每段进行 recursive 二次切分
    """
    pattern = re.compile(r"(?im)^(OBJECTIVE|BACKGROUND|METHODS|RESULTS|CONCLUSIONS)\b[:\t ]*")
    matches = list(pattern.finditer(text))
    if not matches:
        return _split_with_recursive(text, chunk_size, chunk_overlap)

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    if not sections:
        return _split_with_recursive(text, chunk_size, chunk_overlap)

    # 英文摘要更常见句号/分号断句，优先用英文分隔符
    english_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )

    chunks = []
    for section in sections:
        chunks.extend(english_splitter.split_text(section))

    return chunks if chunks else _split_with_recursive(text, chunk_size, chunk_overlap)


def _split_with_markdown_header(text, chunk_size, chunk_overlap):
    """
    基于 Markdown 标题切分。
    对无标题文本会自动回退 recursive，保证可用性。
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except Exception as e:
        logging.warning(f"MarkdownHeaderTextSplitter 不可用，回退 recursive: {str(e)}")
        return _split_with_recursive(text, chunk_size, chunk_overlap)

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = md_splitter.split_text(text)
    if not docs:
        return _split_with_recursive(text, chunk_size, chunk_overlap)

    # 标题切分后做一次递归二次切分，避免单段过长
    chunks = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        chunks.extend(_split_with_recursive(content, chunk_size, chunk_overlap))
    return chunks if chunks else _split_with_recursive(text, chunk_size, chunk_overlap)


def _split_with_llama_nodes(text, chunk_size, chunk_overlap):
    """
    基于 LlamaIndex 节点切分。
    依赖未安装或异常时自动回退 recursive。
    """
    try:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document
    except Exception as e:
        logging.warning(f"LlamaIndex 不可用，回退 recursive: {str(e)}")
        return _split_with_recursive(text, chunk_size, chunk_overlap)

    try:
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        chunks = [n.text.strip() for n in nodes if getattr(n, "text", "").strip()]
        return chunks if chunks else _split_with_recursive(text, chunk_size, chunk_overlap)
    except Exception as e:
        logging.warning(f"LlamaIndex 切分失败，回退 recursive: {str(e)}")
        return _split_with_recursive(text, chunk_size, chunk_overlap)


def split_text(text, chunk_size=None, chunk_overlap=None, strategy=None):
    """
    将长文本切分为多个片段

    使用 RecursiveCharacterTextSplitter 递归切分：
    先尝试按段落分割，若片段仍过大则按句子分割，以此类推。

    Args:
        text: 待切分的长文本
        chunk_size: 每个片段的最大字符数（默认使用配置值 400）
        chunk_overlap: 相邻片段的重叠字符数（默认使用配置值 40）
        strategy: 指定切分策略（可覆盖 CHUNK_STRATEGY）

    Returns:
        切分后的文本片段列表
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    selected_strategy = (strategy or CHUNK_STRATEGY or "recursive").strip().lower()

    if selected_strategy == "structured_recursive":
        return _split_with_structured_recursive(text, chunk_size, chunk_overlap)
    if selected_strategy == "markdown_header":
        return _split_with_markdown_header(text, chunk_size, chunk_overlap)
    if selected_strategy == "llama_nodes":
        return _split_with_llama_nodes(text, chunk_size, chunk_overlap)
    if selected_strategy != "recursive":
        logging.warning(f"未知切分策略 {selected_strategy}，自动回退 recursive")
    return _split_with_recursive(text, chunk_size, chunk_overlap)
