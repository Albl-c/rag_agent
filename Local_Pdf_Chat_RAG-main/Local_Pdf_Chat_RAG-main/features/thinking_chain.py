"""
思维链处理 —— DeepSeek-R1 思维链标签的格式化

学习要点：
- DeepSeek-R1 模型会在回答中输出 <think>...</think> 标签，包含推理过程
- 本模块将思维链内容转换为可折叠的 HTML 详情框
"""

import logging


def process_thinking_content(text):
    """
    处理包含 <think> 标签的内容，将其转换为可折叠的 HTML 格式

    将 <think>推理过程</think> 转换为 <details> 可折叠标签。
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            processed_text = str(text)
        except:
            return "无法处理的内容格式"
    else:
        processed_text = text

    try:
        while "<think>" in processed_text and "</think>" in processed_text:
            start_idx = processed_text.find("<think>")
            end_idx = processed_text.find("</think>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = processed_text[start_idx + 7:end_idx]
                before = processed_text[:start_idx]
                after = processed_text[end_idx + 8:]
                processed_text = (
                    before +
                    "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" +
                    thinking_content +
                    "\n\n</details>\n\n" +
                    after
                )

        # 处理其他 HTML 标签，保留 details 和 summary
        processed_html = []
        i = 0
        while i < len(processed_text):
            if (processed_text[i:i + 8] == "<details" or
                    processed_text[i:i + 9] == "</details" or
                    processed_text[i:i + 8] == "<summary" or
                    processed_text[i:i + 9] == "</summary"):
                tag_end = processed_text.find(">", i)
                if tag_end != -1:
                    processed_html.append(processed_text[i:tag_end + 1])
                    i = tag_end + 1
                    continue
            if processed_text[i] == "<":
                processed_html.append("&lt;")
            elif processed_text[i] == ">":
                processed_html.append("&gt;")
            else:
                processed_html.append(processed_text[i])
            i += 1

        processed_text = "".join(processed_html)
    except Exception as e:
        logging.error(f"处理思维链内容时出错: {str(e)}")
        try:
            return text.replace("<", "&lt;").replace(">", "&gt;")
        except:
            return "处理内容时出错"

    return processed_text
