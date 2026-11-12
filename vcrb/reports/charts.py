"""matplotlib 图表（6 张，中文字体回退，无数据时输出占位图）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..evaluation.scoring import OUTCOME_LABELS, ScoreResult  # noqa: E402

_SLICE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#9D755D", "#EECA3B",
]

# 中文字体回退链（WSL/Linux 常见 Noto/文泉驿；Windows/macOS 回退）
_FONT_CHAIN = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["font.sans-serif"] = _FONT_CHAIN
matplotlib.rcParams["axes.unicode_minus"] = False

PALETTE = {
    "A": "#4C78A8",
    "B": "#F58518",
}


def _model_labels(models: list[str]) -> list[str]:
    return models


def _placeholder(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16, color="#888")
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _has_data(result: ScoreResult) -> bool:
    return any(c.planned > 0 for c in result.metrics.values())


def _accuracy_with_ci(ax, xs, accs, cis, width, colors):
    lows = [max(0.0, a - ci[0]) for a, ci in zip(accs, cis)]
    highs = [max(0.0, ci[1] - a) for a, ci in zip(accs, cis)]
    ax.bar(
        xs, accs, width=width, color=colors, yerr=[lows, highs],
        capsize=3, ecolor="#555", alpha=0.92,
    )
    for x, acc in zip(xs, accs):
        ax.text(x, 0.02, f"{acc * 100:.1f}%", ha="center", fontsize=8, color="white")


def render_charts(result: ScoreResult, out_dir: Path) -> list[Path]:
    """生成 6 张图；返回已生成文件列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _has_data(result):
        paths = []
        names = [
            "overall_accuracy.png",
            "ab_lift.png",
            "by_work.png",
            "by_importance.png",
            "by_difficulty.png",
            "error_types.png",
        ]
        for name in names:
            p = out_dir / name
            _placeholder(p, "待真实实验数据")
            paths.append(p)
        return paths

    models = sorted({m for m, _ in result.metrics})
    written: list[Path] = []

    # 1. 总体准确率（A/B 分组 + 95%CI）
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    xs = range(len(models))
    width = 0.38 if len(models) > 1 else 0.5
    for i, exp in enumerate(("A", "B")):
        accs, cis = [], []
        for m in models:
            cell = result.metrics.get((m, exp))
            accs.append(cell.accuracy_valid if cell else 0.0)
            cis.append(cell.accuracy_valid_ci if cell else (0.0, 0.0))
        offset = (i - 0.5) * width if len(models) > 1 else 0
        _accuracy_with_ci(
            ax, [x + offset for x in xs], accs, cis, width, [PALETTE[exp]] * len(models)
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_model_labels(models), rotation=20, ha="right")
    ax.set_ylabel("有效响应识别准确率")
    ax.set_title("模型总体准确率对比（误差棒=95% Wilson CI）")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(color=PALETTE["A"], label="A组 纯视觉"), Patch(color=PALETTE["B"], label="B组 提供作品名")])
    p = out_dir / "overall_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2. A/B 提升百分点
    lifts = []
    for m in models:
        a = result.metrics.get((m, "A"))
        b = result.metrics.get((m, "B"))
        if a and b:
            lifts.append((m, (b.accuracy_valid - a.accuracy_valid) * 100))
    fig, ax = plt.subplots(figsize=(max(8, len(lifts) * 1.5), 4.5))
    if lifts:
        lifts.sort(key=lambda t: t[1])
        names = [t[0] for t in lifts]
        vals = [t[1] for t in lifts]
        colors = ["#54A24B" if v >= 0 else "#E45756" for v in vals]
        ax.barh(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}pp", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("B组 − A组 准确率提升（百分点）")
    ax.set_title("提供作品名称带来的准确率提升")
    p = out_dir / "ab_lift.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3-5. 分作品 / 分重要度 / 分难度（B组优先展示，A组虚线对照可后续扩展）
    def grouped_bar(slice_dict: dict, title: str, fname: str, order: list[str] | None = None) -> Path:
        keys: list[str] = order or []
        if not keys:
            for m in models:
                for k in slice_dict.get((m, "B"), {}):
                    if k not in keys:
                        keys.append(k)
        colors = [_SLICE_COLORS[i % len(_SLICE_COLORS)] for i in range(len(keys))]
        fig, ax = plt.subplots(figsize=(max(8, len(models) * max(1.2, len(keys) * 0.5)), 5))
        n_groups = max(len(models), 1)
        n_bars = len(keys) or 1
        total_width = 0.8 / max(n_groups, 1)
        bar_width = total_width / n_bars
        for gi, m in enumerate(models):
            cell_map = slice_dict.get((m, "B"), {})
            for ki, k in enumerate(keys):
                cell = cell_map.get(k)
                acc = cell.accuracy_valid if cell else 0.0
                x = gi - 0.4 + bar_width * (ki + 0.5)
                ax.bar(x, acc, width=bar_width * 0.92, color=colors[ki])
        ax.set_xticks([i - 0.4 + total_width / 2 for i in range(n_groups)])
        ax.set_xticklabels(_model_labels(models), rotation=20, ha="right")
        if keys:
            ax.legend(
                handles=[Patch(color=c, label=k) for k, c in zip(keys, colors)],
                fontsize=8, ncol=min(3, len(keys)),
            )
        ax.set_ylabel("有效响应识别准确率")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        p = out_dir / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    written.append(grouped_bar(result.by_work, "分作品准确率（B组）", "by_work.png"))
    written.append(
        grouped_bar(
            result.by_importance,
            "分人物重要度准确率（B组）",
            "by_importance.png",
            order=["main", "key_supporting", "supporting"],
        )
    )
    written.append(
        grouped_bar(
            result.by_difficulty,
            "分视觉难度准确率（B组）",
            "by_difficulty.png",
            order=["normal", "side_face", "occluded", "blurry", "costume_change"],
        )
    )

    # 6. 错误类型分布（堆叠柱）
    error_kinds = ["wrong_name", "actor_name", "uncertain", "refused", "parse_error", "api_failure"]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.7), 5))
    bottoms = [0] * len(models)
    colors = ["#E45756", "#B279A2", "#F2CF5B", "#EECA3B", "#72B7B2", "#9D755D"]
    for kind, color in zip(error_kinds, colors):
        vals = []
        for m in models:
            total = 0
            for exp in ("A", "B"):
                cell = result.metrics.get((m, exp))
                if cell:
                    total += cell.counts.get(kind, 0)
            vals.append(total)
        ax.bar(_model_labels(models), vals, bottom=bottoms, color=color, label=OUTCOME_LABELS[kind])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("样本数（A+B 合计）")
    ax.set_title("错误类型分布")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    p = out_dir / "error_types.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written
