"""本地视频抽帧：场景切换检测（避免连续重复帧）或固定间隔。

依赖 imageio-ffmpeg（pip install "vcrb[video]"，自带 ffmpeg 二进制），
系统已装 ffmpeg 时也可直接使用。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FfmpegNotAvailable(RuntimeError):
    pass


@dataclass
class FrameReport:
    video_path: str
    strategy: str
    frames: int = 0
    errors: list[str] = field(default_factory=list)


def resolve_ffmpeg() -> str:
    """优先系统 ffmpeg，其次 imageio-ffmpeg 自带二进制。"""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise FfmpegNotAvailable(
            "未找到 ffmpeg：请安装系统 ffmpeg 或运行 pip install \"vcrb[video]\""
        ) from exc


_SHOWINFO_RE = re.compile(r"pts_time:([0-9.]+)")


def extract_frames(
    video_path: Path,
    out_dir: Path,
    strategy: str = "scene",
    scene_threshold: float = 0.3,
    interval_seconds: float = 10.0,
    max_frames: int = 200,
) -> FrameReport:
    """抽帧到 out_dir，并写入/追加 manifest.jsonl（source_type=frame_capture）。

    strategy:
      scene    场景切换检测（默认，天然避免连续帧刷屏）
      interval 每 interval_seconds 秒取一帧
    """
    if strategy not in {"scene", "interval"}:
        raise ValueError(f"未知抽帧策略：{strategy}")
    ffmpeg = resolve_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = FrameReport(video_path=str(video_path), strategy=strategy)

    if strategy == "scene":
        vf = f"select='gt(scene,{scene_threshold})',showinfo"
    else:
        vf = f"fps={1.0 / max(interval_seconds, 0.1)},showinfo"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "info",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(out_dir / "frame_%04d.jpg"),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0:
        report.errors.append(proc.stderr[-800:])
        return report

    frames = sorted(out_dir.glob("frame_*.jpg"))
    # 场景策略下 ffmpeg 从 1 编号可能覆盖旧文件——重命名为本视频唯一前缀
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as fh:
        for idx, frame in enumerate(frames, start=1):
            new_name = f"vid_{stamp}_{idx:04d}.jpg"
            frame.rename(out_dir / new_name)
            fh.write(
                json.dumps(
                    {
                        "file": new_name,
                        "source_type": "frame_capture",
                        "source_url": str(video_path),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "license_note": "本地视频文件抽帧（使用者需自行确认素材权利）",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            report.frames += 1
    return report
