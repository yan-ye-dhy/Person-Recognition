"""视觉启发式筛查：只标记「明显异常」，不做身份判断。

可离线检测（PIL + numpy，无预训练模型、无许可负担）：
- grayscale_image：低饱和度 → 黑白漫画截图嫌疑（动画类直接排除依据之一）
- collage_grid：行/列梯度投影出现近似等间距的强分割线 → 拼贴/组图
- poster_layout：竖版高宽比 → 海报嫌疑（needs_review）
- odd_aspect：明显偏离常见正片画幅 → needs_review
- white_border：边缘大面积近白 → 插画/杂志页嫌疑（动画类 needs_review）

局限（如实声明）：Cosplay、同画风二创、AI 生成图无法离线可靠识别，
依赖证据规则与人工批量审核兜底。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class VisualReport:
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.flags


def _downscale(img: Image.Image, max_side: int = 480) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _saturation(arr: np.ndarray) -> np.ndarray:
    """RGB uint8 数组 → [0,1] 饱和度（HSV 的 S 通道）。"""
    f = arr.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    denom = np.where(mx > 0, mx, 1.0)
    return (mx - mn) / denom


def _projection_peaks(profile: np.ndarray, median_ratio: float, min_gap: int) -> list[int]:
    """梯度投影中的显著峰位置（> median × median_ratio，且彼此间隔 ≥ min_gap）。"""
    med = float(np.median(profile)) or 1.0
    thresh = med * median_ratio
    peaks: list[int] = []
    for i, v in enumerate(profile):
        if v <= thresh:
            continue
        if peaks and i - peaks[-1] < min_gap:
            # 段内取最大值：更新为更强峰
            if v > profile[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def _spanning_peaks(diff_map: np.ndarray, axis: int, peaks: list[int], span_min: float) -> list[int]:
    """过滤峰：只保留「贯穿性」结构。

    真拼贴分割线在整列/整行上都有强梯度（span≈1）；人物轮廓、柱子
    边缘只覆盖画面局部（span<0.8），据此排除假分割线。
    diff_map: bool 矩阵（强梯度位置）；axis=1 检查列（竖线），axis=0 检查行。
    """
    out: list[int] = []
    for p in peaks:
        col = diff_map[:, p] if axis == 1 else diff_map[p, :]
        if float(col.mean()) >= span_min:
            out.append(p)
    return out


def _regular_spacing(peaks: list[int], size: int, cv_max: float) -> bool:
    """峰间距是否近似等距（网格特征）。"""
    if len(peaks) < 2:
        return False
    gaps = [b - a for a, b in zip(peaks, peaks[1:])]
    mean = sum(gaps) / len(gaps)
    if mean <= 0 or mean > size * 0.6:  # 单条粗分割线不算网格
        return False
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return (var ** 0.5) / mean <= cv_max


def analyze_image(
    path: Path, cfg: dict, is_animation: bool = False
) -> VisualReport:
    """对单张图片做视觉启发式分析（只产生标记，不判定归属）。"""
    report = VisualReport()
    try:
        with Image.open(path) as im:
            img = _downscale(im.convert("RGB"))
    except OSError:
        report.flags.append("unreadable")
        report.metrics["error"] = 1.0
        return report

    arr = np.asarray(img)
    h, w = arr.shape[:2]
    aspect = w / max(h, 1)
    report.metrics["aspect"] = round(aspect, 3)

    # 1) 灰度/黑白（漫画截图嫌疑）
    sat = _saturation(arr)
    sat_mean = float(sat.mean())
    sat_p95 = float(np.percentile(sat, 95))
    report.metrics["sat_mean"] = round(sat_mean, 4)
    report.metrics["sat_p95"] = round(sat_p95, 4)
    if sat_mean < float(cfg.get("grayscale_sat_mean", 0.05)) and sat_p95 < float(
        cfg.get("grayscale_sat_p95", 0.12)
    ):
        report.flags.append("grayscale_image")

    # 2) 拼贴/组图：行列梯度投影的等距「贯穿性」分割线
    gray = arr.mean(axis=2)
    diff_x = np.abs(np.diff(gray, axis=1))  # (h, w-1) 列方向梯度
    diff_y = np.abs(np.diff(gray, axis=0))  # (h-1, w) 行方向梯度
    gx = diff_x.sum(axis=0)                 # 垂直分割线 → 列能量
    gy = diff_y.sum(axis=1)                 # 水平分割线 → 行能量
    ratio = float(cfg.get("collage_peak_ratio", 3.0))
    span_min = float(cfg.get("collage_span_min", 0.85))
    # 强梯度二值图：用于贯穿性检验（阈值取全图梯度的 P95，防噪声抬升基线）
    t_x = float(np.percentile(diff_x, 95)) or 1.0
    t_y = float(np.percentile(diff_y, 95)) or 1.0
    span_map_x = diff_x > t_x
    span_map_y = diff_y > t_y
    col_peaks = _spanning_peaks(
        span_map_x, 1, _projection_peaks(gx, ratio, min_gap=max(8, w // 12)), span_min
    )
    row_peaks = _spanning_peaks(
        span_map_y, 0, _projection_peaks(gy, ratio, min_gap=max(8, h // 12)), span_min
    )
    cv_max = float(cfg.get("collage_spacing_cv", 0.35))
    grid_cols = _regular_spacing(col_peaks, w, cv_max)
    grid_rows = _regular_spacing(row_peaks, h, cv_max)
    report.metrics["col_peaks"] = len(col_peaks)
    report.metrics["row_peaks"] = len(row_peaks)
    # 单方向 ≥2 条等距贯穿分割线 → 拼贴嫌疑（3 联/4 联条）。
    # 2×2 田字格（每方向仅 1 条线）与局部强边缘（人物轮廓/立柱）不触发，
    # 依赖人工批量审核兜底（如实声明的局限）。
    if (grid_cols and len(col_peaks) >= 2) or (grid_rows and len(row_peaks) >= 2):
        report.flags.append("collage_grid")

    # 3) 竖版海报 / 异常画幅
    if aspect < float(cfg.get("poster_aspect_max", 0.85)):
        report.flags.append("poster_layout")
    elif aspect < float(cfg.get("odd_aspect_min", 1.30)) or aspect > float(
        cfg.get("odd_aspect_max", 2.45)
    ):
        report.flags.append("odd_aspect")

    # 4) 插画/留白（动画类重点）：边缘近白占比
    thresh_v = float(cfg.get("white_border_threshold", 235))
    frac = float(cfg.get("white_border_frac", 0.60))
    band = max(2, h // 24)
    top = arr[:band].astype(np.int16)
    bottom = arr[-band:].astype(np.int16)
    left = arr[:, :max(2, w // 24)].astype(np.int16)
    right = arr[:, -max(2, w // 24):].astype(np.int16)
    strips = np.concatenate([
        top.reshape(-1, 3), bottom.reshape(-1, 3),
        left.reshape(-1, 3), right.reshape(-1, 3),
    ])
    white_frac = float((strips.min(axis=1) > thresh_v).mean())
    report.metrics["white_border_frac"] = round(white_frac, 4)
    if is_animation and white_frac >= frac:
        report.flags.append("white_border")

    return report
