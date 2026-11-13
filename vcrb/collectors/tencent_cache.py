"""从已下载的腾讯 HLS 本地分片中抽取可解码画面。

只读取本地索引及分片，不使用客户端凭据、网络地址或解密密钥。
部分 VVC 分片有损坏/缺失的参考帧，故逐片抽关键帧，并记录解码警告。
本功能不保证其他格式缓存可读取，也不保证可以导出完整视频。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

from ..imageutils import check_image, is_solid_color
from .video_frames import resolve_ffmpeg


@dataclass(frozen=True)
class CacheSegment:
    index: int
    path: Path
    start_seconds: float
    duration_seconds: float


def read_cache_segments(cache_dir: Path) -> list[CacheSegment]:
    """将索引中的 index 映射到本地数字命名的 TS；不请求索引中的 URL。"""
    cache_dir = cache_dir.resolve()
    playlist = cache_dir / ".m3u8"
    if not playlist.is_file():
        raise ValueError("缓存目录中缺少 .m3u8，请指定对应剧集的 .hls 目录")
    files: dict[int, Path] = {}
    for path in cache_dir.rglob("*.ts"):
        if not path.stem.isdigit():
            continue
        if not path.resolve().is_relative_to(cache_dir):
            raise ValueError("分片路径超出指定缓存目录")
        index = int(path.stem)
        if index in files:
            raise ValueError(f"存在多个编号为 {index} 的分片，无法可靠匹配")
        files[index] = path

    segments = []
    elapsed = 0.0
    duration = None
    seen: set[int] = set()
    for line in playlist.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            duration = float(line.partition(":")[2].partition(",")[0])
            if duration <= 0:
                raise ValueError("分片时长必须大于零")
        elif line and not line.startswith("#"):
            query = parse_qs(urlsplit(line).query)
            try:
                index = int(query["index"][0])
            except (KeyError, ValueError) as exc:
                raise ValueError("缓存索引缺少数字 index，暂不支持此格式") from exc
            if duration is None or index in seen:
                raise ValueError("缓存索引缺少时长或含重复分片编号")
            if index not in files:
                raise ValueError(f"本地分片 {index} 缺失，请等待下载完成")
            segments.append(CacheSegment(index, files[index], elapsed, duration))
            seen.add(index)
            elapsed += duration
            duration = None
    if not segments:
        raise ValueError("缓存索引中没有可用分片")
    return segments


@dataclass
class CacheFrameReport:
    selected: int = 0
    frames: int = 0
    skipped_existing: int = 0
    decode_warnings: int = 0
    errors: list[str] = field(default_factory=list)


def extract_cache_frames(
    cache_dir: Path,
    out_dir: Path,
    *,
    max_frames: int = 100,
    skip_start_seconds: float = 180,
    skip_end_seconds: float = 120,
    title: str = "",
) -> CacheFrameReport:
    """分散选取本地分片，每片最多一张；原始尺寸，无缩放，供人工复核。"""
    if max_frames < 1 or skip_start_seconds < 0 or skip_end_seconds < 0:
        raise ValueError("采样数量必须为正，跳过时长不能为负")
    segments = read_cache_segments(cache_dir)
    end = segments[-1].start_seconds + segments[-1].duration_seconds - skip_end_seconds
    available = [
        s for s in segments
        if s.start_seconds >= skip_start_seconds
        and s.start_seconds + s.duration_seconds <= end
    ]
    if not available:
        raise ValueError("跳过片头片尾后无可用分片，请调整 --skip-start / --skip-end")
    count = min(max_frames, len(available))
    positions = [len(available) // 2] if count == 1 else [
        i * (len(available) - 1) // (count - 1) for i in range(count)
    ]
    selected = [available[i] for i in positions]
    report = CacheFrameReport(selected=len(selected))
    ffmpeg = resolve_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    known = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line)["file"])
    source_id = hashlib.sha256(str(cache_dir.resolve()).encode()).hexdigest()[:12]
    with TemporaryDirectory(prefix=".cache-frame-", dir=out_dir) as temp:
        for segment in selected:
            frame = Path(temp) / f"{segment.index}.jpg"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "info", "-nostdin",
                # 可以输出有效关键帧时，仍将其余丢帧警告记录供审核。
                "-max_error_rate", "1", "-skip_frame", "nokey",
                "-err_detect", "explode", "-fflags", "+discardcorrupt",
                "-f", "mpegts", "-i", str(segment.path),
                "-an", "-vf", "showinfo", "-frames:v", "1", "-q:v", "2", str(frame),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            except subprocess.TimeoutExpired:
                report.errors.append(f"分片 {segment.index} 解码超时")
                continue
            if proc.returncode or not check_image(frame, 300, 200).valid:
                report.errors.append(f"分片 {segment.index} 未能输出有效图片")
                continue
            if is_solid_color(frame):
                report.errors.append(f"分片 {segment.index} 输出空白/纯色画面")
                continue
            pts = re.search(r"pts_time:([0-9.]+)", proc.stderr)
            if pts is None:
                report.errors.append(f"分片 {segment.index} 缺少画面时间戳")
                continue
            timestamp = segment.start_seconds + float(pts[1])
            if timestamp >= end:
                report.errors.append(f"分片 {segment.index} 输出画面超出指定时间范围")
                continue
            warning = any(t in proc.stderr for t in ("Error", "Failed", "corrupt"))
            report.decode_warnings += int(warning)
            digest = hashlib.md5(frame.read_bytes()).hexdigest()
            name = f"cache_{source_id}_{segment.index:04d}_{digest[:8]}.jpg"
            if name in known and (out_dir / name).is_file():
                report.skipped_existing += 1
                continue
            frame.replace(out_dir / name)
            entry = {
                "file": name, "source_type": "frame_capture",
                "source_url": f"{cache_dir.resolve()}#t={timestamp:.3f}",
                "page_title": title, "source_channel": "腾讯视频客户端下载缓存",
                "video_timestamp_seconds": round(timestamp, 3),
                "segment_index": segment.index,
                "segment_start_seconds": segment.start_seconds,
                "timestamp_basis": "playlist_segment_start_plus_decoder_pts",
                "decode_warnings": warning, "requires_visual_review": True,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license_note": "使用者提供的本地缓存抽帧，角色与素材质量待审核",
                "md5": digest,
            }
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            known.add(name)
            report.frames += 1
    return report
