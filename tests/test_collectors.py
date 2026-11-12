"""采集器测试：来源解析、下载（mock transport）、抽帧参数校验。"""

from __future__ import annotations

import json

import httpx
import pytest

from vcrb.collectors.sources import (
    BingImageSource,
    UrlListSource,
    source_type_from_query,
)
from vcrb.collectors.web_collector import collect_web, download_hits
from vcrb.collectors.video_frames import extract_frames
from vcrb.config import load_settings
from tests.conftest import make_work


def test_bing_parse_html():
    src = BingImageSource(user_agent="test")
    html = (
        '<div class="iusc" m=\'{"t":"《狂飙》剧照","murl":"https://a.com/1.jpg",'
        '"purl":"https://www.iqiyi.com/a.html"}\'></div>'
        '<div class="iusc" m=\'{"murl":"https://b.com/2.png"}\'></div>'
        '<div class="iusc" m=\'{"murl":"https://a.com/1.jpg"}\'></div>'
    )
    hits = src.parse_html(html)
    assert [h.url for h in hits] == ["https://a.com/1.jpg", "https://b.com/2.png"]
    assert all(h.source_type == "web_image" for h in hits)
    # 归属证据随命中带出（页面标题/页面 URL）
    assert hits[0].page_title == "《狂飙》剧照"
    assert hits[0].page_url == "https://www.iqiyi.com/a.html"
    assert hits[1].page_title == "" and hits[1].page_url == ""


def test_bing_search_network_blocked_returns_empty():
    src = BingImageSource(user_agent="test", timeout=0.001)
    assert src.search("狂飙 剧照") == []


def test_source_type_from_query():
    assert source_type_from_query("狂飙 电视剧 剧照 高启强") == "official_still"
    assert source_type_from_query("剑来 动漫 截图 角色") == "episode_screenshot"
    assert source_type_from_query("剑来") == "web_image"


def test_url_list_source(tmp_path):
    list_file = tmp_path / "jianlai.txt"
    list_file.write_text(
        "# 注释\nhttps://x.com/1.jpg\n"
        '{"url": "https://y.com/2.jpg", "source_type": "official_still"}\n\n',
        encoding="utf-8",
    )
    hits = UrlListSource(list_file).hits()
    assert len(hits) == 2
    assert hits[0].url == "https://x.com/1.jpg"
    assert hits[1].source_type == "official_still"
    assert UrlListSource(tmp_path / "missing.txt").hits() == []


def _fake_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_download_hits_filters(tmp_path):
    big = b"\xff\xd8" + b"x" * 20000
    small = b"\xff\xd8" + b"x" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "ok" in url:
            return httpx.Response(200, content=big, headers={"content-type": "image/jpeg"})
        if "dup" in url:
            return httpx.Response(200, content=big, headers={"content-type": "image/jpeg"})
        if "small" in url:
            return httpx.Response(200, content=small, headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    from vcrb.collectors.web_collector import CollectReport

    report = CollectReport(work_slug="test")
    def _hit(url):
        return type("H", (), {"url": url, "source_type": "web_image",
                              "license_note": "", "referer": "",
                              "page_url": "", "page_title": "狂飙 页面"})()

    hits = [_hit(u) for u in
            ("https://ok/1.jpg", "https://dup/2.jpg", "https://small/3.jpg", "https://dead/4.jpg")]
    download_hits(
        hits,
        raw_dir=tmp_path / "raw",
        user_agent="test",
        request_timeout=5,
        min_image_bytes=1024,
        max_total=10,
        report=report,
        client=_fake_client(handler),
    )
    assert report.downloaded == 1   # ok 成功；dup 同 md5 跳过
    assert report.skipped_duplicate == 1
    assert report.skipped_small == 1
    assert report.failed == 1
    manifest = [
        json.loads(line)
        for line in (tmp_path / "raw" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 1
    assert manifest[0]["source_type"] == "web_image"
    assert manifest[0]["md5"]


def test_collect_web_with_stubbed_search(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        # 每个 URL 返回不同内容，避免 md5 去重干扰上限测试
        marker = abs(hash(str(request.url))) % 250
        return httpx.Response(
            200,
            content=b"\xff\xd8" + bytes([marker]) * 20000,
            headers={"content-type": "image/jpeg"},
        )

    def fake_search(self, query, limit=35):
        # 第一条命中带作品标识标题（可下载）；第二条无任何证据（默认被预检跳过）
        qh = abs(hash(query)) % 1000
        with_ev = type("H", (), {
            "url": f"https://img/{qh}.jpg", "source_type": "web_image",
            "license_note": "", "referer": "",
            "page_url": "https://www.iqiyi.com/lib/k.html", "page_title": "狂飙 电视剧 剧照",
        })()
        no_ev = type("H", (), {
            "url": f"https://img/noev-{qh}.jpg", "source_type": "web_image",
            "license_note": "", "referer": "", "page_url": "", "page_title": "功夫足球",
        })()
        return [with_ev, no_ev]

    monkeypatch.setattr(BingImageSource, "search", fake_search)
    settings = load_settings()
    settings["collect"]["request_timeout_seconds"] = 5
    settings["collect"]["min_image_bytes"] = 1024
    work = make_work("kuangbiao")
    report = collect_web(
        work, settings, raw_dir=tmp_path / "raw", url_list_dir=tmp_path / "lists",
        limit=5, client=_fake_client(handler),
    )
    # 工厂作品带 3 个查询词，每个查询 2 条命中（1 条有作品标识 + 1 条无证据）
    assert report.downloaded == 3
    assert report.skipped_no_evidence == 3
    manifest = [
        json.loads(line)
        for line in (tmp_path / "raw" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 3
    # 剧照查询 → official_still 溯源
    assert any(m["source_type"] == "official_still" for m in manifest)


def test_extract_frames_rejects_bad_strategy(tmp_path):
    with pytest.raises(ValueError):
        extract_frames(tmp_path / "v.mp4", tmp_path / "out", strategy="everyframe")
