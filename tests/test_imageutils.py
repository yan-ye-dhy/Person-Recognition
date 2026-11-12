"""图片工具测试。"""

from __future__ import annotations

from PIL import Image

from vcrb.imageutils import (
    check_image,
    draw_target_box,
    encode_for_request,
    is_solid_color,
    strip_and_normalize,
)


def test_check_image_valid(make_image):
    path = make_image(size=(640, 360))
    check = check_image(path, min_width=300, min_height=200)
    assert check.valid
    assert (check.width, check.height) == (640, 360)
    assert check.format == "JPEG"


def test_check_image_too_small(make_image):
    path = make_image(size=(200, 100))
    check = check_image(path, min_width=300, min_height=200)
    assert not check.valid
    assert "尺寸过小" in check.reason


def test_check_image_corrupt(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    check = check_image(path)
    assert not check.valid


def test_check_image_bad_suffix(tmp_path):
    path = tmp_path / "image.gif"
    path.write_bytes(b"whatever")
    check = check_image(path)
    assert not check.valid
    assert "扩展名" in check.reason


def test_draw_target_box_pixels(make_image, tmp_path):
    src = make_image(size=(400, 300), color=(0, 0, 0))
    dst = tmp_path / "marked.jpg"
    draw_target_box(src, dst, [0.25, 0.25, 0.75, 0.75])
    assert dst.exists()
    img = Image.open(dst)
    assert img.size == (400, 300)
    # 框线应为红色：检查顶部边框中点
    pixel = img.getpixel((200, 75))
    assert pixel[0] > 180 and pixel[1] < 80 and pixel[2] < 80


def test_draw_target_box_none_bbox(make_image, tmp_path):
    src = make_image()
    dst = tmp_path / "unmarked.jpg"
    draw_target_box(src, dst, None)
    assert dst.exists()


def test_encode_for_request_prefix_and_meta(make_image):
    path = make_image(size=(800, 600))
    data_url, meta = encode_for_request(path)
    assert data_url.startswith("data:image/jpeg;base64,")
    assert meta["original_size"] == [800, 600]
    assert meta["encoded_format"] == "JPEG"


def test_encode_for_request_downscales(make_image):
    path = make_image(size=(3200, 1800))
    data_url, meta = encode_for_request(path, max_side=1600)
    assert max(meta["encoded_size"]) <= 1600
    assert len(data_url) < 1_500_000  # 大图也不至于爆请求体


def test_is_solid_color(make_image, tmp_path):
    solid = make_image(name="solid.jpg", size=(400, 300), color=(10, 20, 30))
    assert is_solid_color(solid)
    noisy = tmp_path / "noisy.jpg"
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    for x in range(0, 400, 40):
        for y in range(0, 300, 40):
            img.putpixel((x, y), (255, 255, 255))
    noisy.write_bytes(b"")
    img.save(noisy, format="JPEG")
    assert not is_solid_color(noisy)


def test_strip_and_normalize_removes_metadata(make_image):
    src = make_image(size=(500, 400), fmt="PNG")
    dst = src.with_suffix(".jpg")
    strip_and_normalize(src, dst)
    assert dst.exists()
    assert dst.suffix == ".jpg"
    assert Image.open(dst).size == (500, 400)
