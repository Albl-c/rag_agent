"""
对比两份评测结果（A/B）并输出关键指标变化。

用法：
python eval/compare_eval.py --base eval/results/ab_fixed.json --target eval/results/ab_route.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f}%"


def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _analyze(base: Dict[str, Any], target: Dict[str, Any], top_k: int) -> str:
    lines: List[str] = []
    b_hit = float(base.get("hit_at_k", 0.0))
    t_hit = float(target.get("hit_at_k", 0.0))
    b_mrr = float(base.get("mrr", 0.0))
    t_mrr = float(target.get("mrr", 0.0))
    b_lat = float(base.get("avg_latency_ms", base.get("avg_retrieval_ms", 0.0)))
    t_lat = float(target.get("avg_latency_ms", target.get("avg_retrieval_ms", 0.0)))

    b_chunk = int(base.get("experiment", {}).get("chunk_count", 0))
    t_chunk = int(target.get("experiment", {}).get("chunk_count", 0))

    lines.append("=== A/B 指标对比 ===")
    lines.append(f"Hit@{top_k}: {_fmt_num(b_hit)} -> {_fmt_num(t_hit)} ({_pct(t_hit - b_hit)})")
    lines.append(f"MRR: {_fmt_num(b_mrr)} -> {_fmt_num(t_mrr)} ({_pct(t_mrr - b_mrr)})")
    lat_label = "AvgLatencyMs" if ("avg_latency_ms" in base or "avg_latency_ms" in target) else "AvgRetrievalMs"
    lines.append(f"{lat_label}: {b_lat:.2f} -> {t_lat:.2f} ({_pct((t_lat - b_lat) / max(1e-9, b_lat))})")
    if b_chunk > 0:
        chunk_delta_ratio = (t_chunk - b_chunk) / b_chunk
    else:
        chunk_delta_ratio = 0.0
    lines.append(f"ChunkCount: {b_chunk} -> {t_chunk} ({_pct(chunk_delta_ratio)})")

    lines.append("\n=== 结论 ===")
    hit_up = t_hit >= b_hit
    mrr_up = t_mrr >= b_mrr
    latency_worse = t_lat > b_lat * 1.15 if b_lat > 0 else False
    chunk_blow_up = t_chunk > b_chunk * 1.5 if b_chunk > 0 else False

    if hit_up and mrr_up and not latency_worse and not chunk_blow_up:
        lines.append("路由优化整体收益为正：召回/排序不降，且时延与分块规模可接受。")
    elif hit_up or mrr_up:
        lines.append("路由优化存在局部收益，但需要继续平衡时延或分块规模。")
    else:
        lines.append("当前路由配置未体现收益，建议回退默认策略或调整路由阈值。")

    if latency_worse:
        lines.append("- 警告：平均时延上升超过 15%。")
    if chunk_blow_up:
        lines.append("- 警告：chunk 数量膨胀超过 50%，会增加索引和检索成本。")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两份 RAG 评测结果")
    parser.add_argument("--base", required=True, help="基线结果文件（如 auto-route=false）")
    parser.add_argument("--target", required=True, help="目标结果文件（如 auto-route=true）")
    args = parser.parse_args()

    base = _load_json(args.base)
    target = _load_json(args.target)
    top_k = int(target.get("experiment", {}).get("top_k", base.get("experiment", {}).get("top_k", 5)))
    print(_analyze(base, target, top_k=top_k))


if __name__ == "__main__":
    main()
