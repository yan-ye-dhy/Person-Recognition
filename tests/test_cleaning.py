"""清洗流水线测试：有效性/去重/质量。"""

from __future__ import annotations

import json
import random

from PIL import Image

from vcrb.dataset.cleaning import (
    clean_work,
    hamming_distance,
    is_near_duplicate,
    load_manifest,
)

SETTINGS = {
    "clean": {
        "min_width": 300,
        "min_height": 200,
        "phash_hamming_threshold": 4,
        "allowed_formats": ["jpg", "jpeg", "png", "webp"],
    }
}


def _write_noise(path, seed: int, size=(640, 360), quality=92):
    """确定性噪声图：不同 seed 的 phash 距离实测 ≥26（阈值 4 之上）。"""
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(0, size[1], 3):
        for x in range(0, size[0], 3):
            color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for dy in range(3):
                for dx in range(3):
                    if x + dx < size[0] and y + dy < size[1]:
                        px[x + dx, y + dy] = color
    img.save(path, format="JPEG", quality=quality)


def _write_reencode(src_path, dst_path, quality=60):
    """同内容重编码（真实场景：同一截图以不同压缩率重复出现 → phash 近重复）。"""
    Image.open(src_path).save(dst_path, format="JPEG", quality=quality)


def test_hamming_distance():
    assert hamming_distance("0" * 16, "0" * 16) == 0
    assert hamming_distance("0" * 16, "1" * 16) == 16  # 0x1111... → 16 个置位位
    # 每个半字节 4 位全不同 → 16×4 = 64
    assert hamming_distance("0f0f0f0f0f0f0f0f", "f0f0f0f0f0f0f0f0") == 64


def test_is_near_duplicate():
    assert is_near_duplicate("0" * 16, "0" * 16, 4)
    assert not is_near_duplicate("0" * 16, "f" * 16, 4)


def test_clean_work_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "candidates"

    _write_noise(raw / "a1.jpg", seed=1)                              # 有效 A（噪声 seed 1）
    (raw / "a2.jpg").write_bytes((raw / "a1.jpg").read_bytes())       # 精确重复（同字节）
    _write_noise(raw / "b1.jpg", seed=2)                              # 有效 B（噪声 seed 2，差异大）
    Image.new("RGB", (640, 360), (5, 5, 5)).save(raw / "solid.jpg")        # 纯色
    Image.new("RGB", (100, 80), (90, 90, 200)).save(raw / "tiny.jpg")      # 尺寸过小
    (raw / "broken.jpg").write_bytes(b"not image")                          # 损坏
    with (raw / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"file": "a1.jpg", "source_type": "official_still", "source_url": "https://x/a1"}) + "\n")
        fh.write(json.dumps({"file": "b1.jpg", "source_type": "web_image"}) + "\n")

    report = clean_work("kuangbiao", raw, out, SETTINGS)

    assert report.total == 6
    assert report.kept == 2
    assert report.exact_duplicates == 1
    assert report.quality_rejected == 1
    assert report.invalid == 2  # 损坏 + 尺寸过小
    files = sorted(p.name for p in out.glob("cand_*"))
    assert files == ["cand_0001.jpg", "cand_0002.jpg"]
    manifest = load_manifest(out)
    assert manifest["cand_0001.jpg"]["source_type"] == "official_still"
    assert manifest["cand_0002.jpg"]["source_type"] == "web_image"
    assert manifest["cand_0001.jpg"]["original_file"] == "a1.jpg"


def test_clean_work_missing_dir(tmp_path):
    report = clean_work("kuangbiao", tmp_path / "nope", tmp_path / "out", SETTINGS)
    assert report.total == 0 and report.kept == 0


def test_near_duplicate_phash(tmp_path):
    """同内容重编码应判近重复；不同内容的噪声图则保留。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_noise(raw / "p1.jpg", seed=3, quality=92)
    _write_reencode(raw / "p1.jpg", raw / "p2.jpg", quality=60)  # 同图重压缩
    _write_noise(raw / "p3.jpg", seed=4)                          # 不同内容
    report = clean_work("jianlai", raw, tmp_path / "out", SETTINGS)
    # p1 保留、p2 近重复剔除、p3 差异大保留
    assert report.kept == 2
    assert report.near_duplicates == 1
    assert report.near_duplicates == 1
