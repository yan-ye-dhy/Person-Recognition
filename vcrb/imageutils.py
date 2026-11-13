"""图片工具：有效性校验、统一预处理（请求编码）、目标框绘制、EXIF 清除。

公平性关键点：
- 所有模型收到的是同一张「红框标注图」，由本模块同一函数生成；
- 请求编码统一为 JPEG data-URL（超长边压缩到 max_side，避免个别厂商因
  分辨率差异获得信息量不同的输入），EXIF 在重编码时天然被清除。
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# 请求图片统一长边上限（四家文档建议范围之内，Kimi 推荐 ≤4K，此处更保守）
REQUEST_MAX_SIDE = 1600
REQUEST_JPEG_QUALITY = 92


@dataclass
class ImageCheck:
    valid: bool
    width: int = 0
    height: int = 0
    format: str = ""
    reason: str = ""


def check_image(path: Path, min_width: int = 0, min_height: int = 0) -> ImageCheck:
    """文件有效性 + 基本尺寸检查（不修改文件）。"""
    if not path.is_file():
        return ImageCheck(False, reason="文件不存在")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return ImageCheck(False, reason=f"不支持的扩展名 {path.suffix}")
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
            img.verify()  # 校验数据完整性
    except (UnidentifiedImageError, OSError) as exc:
        return ImageCheck(False, reason=f"无法解析: {type(exc).__name__}")
    if width < min_width or height < min_height:
        return ImageCheck(
            False,
            width=width,
            height=height,
            format=fmt,
            reason=f"尺寸过小 {width}x{height}（要求 ≥{min_width}x{min_height}）",
        )
    return ImageCheck(True, width=width, height=height, format=fmt)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def draw_target_box(src: Path, dst: Path, bbox: list[float] | None) -> Path:
    """在图片上绘制红色目标框（归一化坐标 [x1,y1,x2,y2]），保存为 dst。

    bbox 为 None 时视为「画面中唯一人物/目标明确」，仍绘制整幅虚框以保持
    A/B 两组与所有模型的输入处理完全一致？——不：为避免无意义标记，
    bbox 为 None 时不画框，提示词侧统一说明"若画面无红框则识别画面主体人物"。
    """
    img = load_rgb(src)
    if bbox:
        x1, y1, x2, y2 = bbox
        w, h = img.size
        left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        thickness = max(3, w // 200)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        for offset in range(thickness):
            draw.rectangle(
                [left - offset, top - offset, right + offset, bottom + offset],
                outline=(255, 0, 0),
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=REQUEST_JPEG_QUALITY)
    return dst


def draw_target_boxes(src: Path, dst: Path, persons) -> Path:
    """Draw numbered targets only; identity labels never enter model images.

    The v1 single-box function is deliberately unchanged. Source bytes are never
    overwritten. Bad boxes and duplicate identifiers fail before writing output.
    """
    from PIL import ImageDraw, ImageFont

    if src.resolve() == dst.resolve():
        raise ValueError("标注图不能覆盖原图")
    targets, seen = [], set()
    for person in persons:
        pid = person.get("person_id") if isinstance(person, dict) else person.person_id
        bbox = person.get("bbox") if isinstance(person, dict) else person.bbox
        if type(pid) is not int or pid < 1 or pid in seen:
            raise ValueError("人物编号必须是唯一正整数")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
            raise ValueError("bbox需为归一化[x1,y1,x2,y2]")
        seen.add(pid)
        targets.append((pid, bbox))
    img = load_rgb(src)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    thickness, size = max(3, w // 200), max(18, round(min(w, h) * .045))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Pillow 10.0 lacks the optional size argument.
            font = ImageFont.load_default()
    occupied = []
    for pid, bbox in sorted(targets):
        x1, y1, x2, y2 = (round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=thickness)
        label = str(pid)
        bounds = draw.textbbox((0, 0), label, font=font)
        bw, bh = bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 12
        # Prefer the lower edge (away from faces), then outside the top edge.
        candidates = [(x1, y2 + 2), (x1, y1 - bh - 2),
                      (x1 + thickness, y2 - bh - thickness), (x2 - bw - thickness, y2 - bh - thickness)]
        choices = []
        for x, y in candidates:
            x, y = max(0, min(w - bw, x)), max(0, min(h - bh, y))
            rect = (x, y, x + bw, y + bh)
            overlap = sum(max(0, min(rect[2], r[2]) - max(rect[0], r[0])) *
                          max(0, min(rect[3], r[3]) - max(rect[1], r[1])) for r in occupied)
            choices.append((overlap, rect))
        _, rect = min(choices, key=lambda item: item[0])
        occupied.append(rect)
        draw.rectangle(rect, fill="red")
        draw.text((rect[0] + 6 - bounds[0], rect[1] + 6 - bounds[1]), label, font=font, fill="white")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=REQUEST_JPEG_QUALITY)
    return dst


def encode_for_request(path: Path, max_side: int = REQUEST_MAX_SIDE) -> tuple[str, dict[str, Any]]:
    """读取图片 → 统一 JPEG 编码 → data-URL。返回 (data_url, 处理元数据)。

    元数据随 run 记录，保证不同模型之间输入预处理可追溯且一致。
    """
    img = load_rgb(path)
    orig_w, orig_h = img.size
    scale = max_side / max(orig_w, orig_h)
    if scale < 1.0:
        new_size = (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=REQUEST_JPEG_QUALITY)  # 重编码即清除 EXIF
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    meta = {
        "original_size": [orig_w, orig_h],
        "encoded_size": list(img.size),
        "encoded_format": "JPEG",
        "quality": REQUEST_JPEG_QUALITY,
        "max_side": max_side,
    }
    return data_url, meta


def strip_and_normalize(src: Path, dst: Path, quality: int = 95) -> Path:
    """重存为 JPEG（清除 EXIF/元数据），用于导出匿名化正式集。"""
    img = load_rgb(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=quality)
    return dst


def is_solid_color(path: Path) -> bool:
    """近乎纯色图（截图黑屏/占位图）检测。"""
    img = load_rgb(path).resize((64, 64))
    extrema = img.getextrema()  # ((min,max)x3)
    return all(lo == hi for lo, hi in extrema)
