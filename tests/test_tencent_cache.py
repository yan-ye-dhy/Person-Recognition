"""本地缓存索引映射、采样溯源和解码失败隔离（不使用客户端或联网）。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from vcrb.collectors import tencent_cache


def _cache(tmp_path: Path, count: int = 5) -> Path:
    root = tmp_path / "episode.hls"
    (root / "segments").mkdir(parents=True)
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for index in range(count):
        (root / "segments" / f"{index}.ts").write_bytes(b"mock local media")
        lines.extend([
            "#EXTINF:10.0,",
            f"unusable_filename.ts?index={index}&token=do-not-persist",
        ])
    (root / ".m3u8").write_text("\n".join(lines) + "\n#EXT-X-ENDLIST\n")
    return root


def test_cache_maps_query_indices_to_local_files_and_keeps_timeline(tmp_path):
    root = _cache(tmp_path)
    segments = tencent_cache.read_cache_segments(root)
    assert [s.start_seconds for s in segments] == [0, 10, 20, 30, 40]
    assert all(s.path.is_file() for s in segments)
    assert [s.path.name for s in segments] == [f"{i}.ts" for i in range(5)]


def test_cache_missing_segment_fails_before_extraction(tmp_path):
    root = _cache(tmp_path)
    (root / "segments/2.ts").unlink()
    with pytest.raises(ValueError, match="分片 2 缺失"):
        tencent_cache.read_cache_segments(root)


def test_cache_sampling_records_time_resolution_and_deduplicates_reruns(tmp_path, monkeypatch):
    root = _cache(tmp_path)
    out = tmp_path / "frames"
    monkeypatch.setattr(tencent_cache, "resolve_ffmpeg", lambda: "test-ffmpeg")

    def decode(cmd, **kwargs):
        image = Image.new("RGB", (640, 360), (20, 30, 60))
        ImageDraw.Draw(image).rectangle((0, 0, 320, 360), fill=(180, 200, 90))
        image.save(cmd[-1])
        return SimpleNamespace(returncode=0, stderr="n: 0 pts_time:0.04 Error submitting packet")

    monkeypatch.setattr(tencent_cache.subprocess, "run", decode)
    options = dict(max_frames=3, skip_start_seconds=0, skip_end_seconds=0, title="剑来 第二季 第04话")
    report = tencent_cache.extract_cache_frames(root, out, **options)
    assert report.frames == 3 and report.decode_warnings == 3
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["segment_index"] for r in rows] == [0, 2, 4]
    assert [r["video_timestamp_seconds"] for r in rows] == [0.04, 20.04, 40.04]
    assert all(r["requires_visual_review"] and r["decode_warnings"] for r in rows)
    assert all(r["page_title"] == options["title"] for r in rows)
    assert "do-not-persist" not in (out / "manifest.jsonl").read_text()
    with Image.open(out / rows[0]["file"]) as image:
        assert image.size == (640, 360)
    again = tencent_cache.extract_cache_frames(root, out, **options)
    assert again.frames == 0 and again.skipped_existing == 3
    assert len((out / "manifest.jsonl").read_text().splitlines()) == 3


def test_failed_decoder_output_never_becomes_a_sample(tmp_path, monkeypatch):
    root = _cache(tmp_path)
    out = tmp_path / "frames"
    monkeypatch.setattr(tencent_cache, "resolve_ffmpeg", lambda: "test-ffmpeg")
    monkeypatch.setattr(
        tencent_cache.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="cannot decode"),
    )
    report = tencent_cache.extract_cache_frames(
        root, out, max_frames=2, skip_start_seconds=0, skip_end_seconds=0,
    )
    assert report.frames == 0 and len(report.errors) == 2
    assert not list(out.glob("*.jpg"))
    assert not (out / "manifest.jsonl").exists()
