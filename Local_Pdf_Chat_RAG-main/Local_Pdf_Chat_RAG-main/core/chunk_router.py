"""
文本切分策略路由器

作用：
1) 基于文件后缀与文本特征，自动选择最合适的切分策略
2) 输出可追踪的路由信息，便于评测与问题排查
"""

import os
import re
from typing import Dict, Tuple


_STRUCTURED_LABEL_PATTERN = re.compile(
    r"(?im)^(OBJECTIVE|BACKGROUND|METHODS|RESULTS|CONCLUSIONS)\b[:\t ]*"
)
_MARKDOWN_HEADER_PATTERN = re.compile(r"(?m)^#{1,3}\s+\S+")


def detect_doc_features(file_name: str, text: str) -> Dict[str, int]:
    """
    检测文本特征并返回计数信息。
    """
    ext = os.path.splitext(file_name or "")[1].lower()
    content = text or ""
    header_count = len(_MARKDOWN_HEADER_PATTERN.findall(content))
    structured_label_count = len(_STRUCTURED_LABEL_PATTERN.findall(content))

    return {
        "ext": ext,
        "char_count": len(content),
        "line_count": content.count("\n") + 1 if content else 0,
        "markdown_header_count": header_count,
        "structured_label_count": structured_label_count,
    }


def route_chunk_strategy(file_name: str, text: str, prefer_llama: bool = False) -> Tuple[str, Dict[str, int]]:
    """
    自动路由文本切分策略。

    规则顺序（命中即返回）：
    1. markdown_header: .md 文件且检测到 >=2 个标题
    2. structured_recursive: 检测到 >=2 个结构化摘要标签
    3. llama_nodes: 仅在 prefer_llama=True 且文本较长时启用
    4. recursive: 兜底
    """
    features = detect_doc_features(file_name=file_name, text=text)

    if features["ext"] == ".md" and features["markdown_header_count"] >= 2:
        return "markdown_header", features

    if features["structured_label_count"] >= 2:
        return "structured_recursive", features

    if prefer_llama and features["char_count"] >= 3000:
        return "llama_nodes", features

    return "recursive", features
