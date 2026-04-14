"""
矛盾检测 —— 检测多来源信息的冲突

学习要点：
- 当 RAG 系统同时使用本地文档和网络搜索时，不同来源可能存在矛盾
- 矛盾检测帮助 LLM 在回答中标注差异，提高回答可信度
"""

import re


def detect_conflicts(sources):
    """检测多来源信息中的矛盾"""
    key_facts = {}
    for item in sources:
        facts = _extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
        for fact, value in facts.items():
            if fact in key_facts and key_facts[fact] != value:
                return True
            key_facts[fact] = value
    return False


def _extract_facts(text):
    """从文本提取关键事实"""
    facts = {}
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    if "产业图谱" in text:
        facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
    return facts


def evaluate_source_credibility(source):
    """评估来源可信度（基于域名简单规则）"""
    credibility_scores = {
        "gov.cn": 0.9, "edu.cn": 0.85, "weixin": 0.7, "zhihu": 0.6, "baidu": 0.5
    }
    url = source.get('url', '')
    if not url:
        return 0.5
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    domain = domain_match.group(1)
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    return 0.5
