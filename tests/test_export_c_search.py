"""Offline report checks: real denominators, search proof and input separation."""
from copy import deepcopy
from html.parser import HTMLParser
import hashlib
import io
import json
import re

from PIL import Image
import pytest

from scripts import export_c_search as report
from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher


class HtmlNode:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return ''.join(child.text() if isinstance(child, HtmlNode) else child for child in self.children)

    def find_all(self, tag=None, class_name=None):
        result = []
        if (tag is None or self.tag == tag) and (class_name is None or class_name in self.attrs.get('class', '').split()):
            result.append(self)
        for child in self.children:
            if isinstance(child, HtmlNode):
                result.extend(child.find_all(tag, class_name))
        return result


class HtmlTree(HTMLParser):
    """Inspect content relationships without depending on layout CSS strings."""
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, html):
        super().__init__()
        self.root = HtmlNode('document')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def table_rows(html):
    tables = HtmlTree(html).root.find_all('table')
    assert len(tables) == 1
    rows = tables[0].find_all('tr')
    return [[cell for cell in row.children if isinstance(cell, HtmlNode) and cell.tag in ('th', 'td')]
            for row in rows]


def raw(names=("安欣",), *, ok=True, metadata=None, invalid_box=False):
    text = json.dumps({"persons": [{"character": name, "bbox": [10, 20, 600, 900] if invalid_box else
                                    [index / len(names), 0, (index + 1) / len(names), 1]}
                                  for index, name in enumerate(names)]}, ensure_ascii=False)
    return {"request": {"model": "test", "system": "不要回答演员姓名", "user": "图片出自测试作品", "image_meta": {"annotation_mode": "unmarked"}, "search_requested": metadata is not None},
            "response": {"ok": ok, "raw_text": text, "search_metadata": metadata or {}, "usage": {}}}


def context(count=2):
    works = {"kuangbiao": "狂飙", "gongfunvzu": "功夫女足", "jianlai": "剑来", "qingyunian2": "庆余年第二季"}
    rosters = {slug: Roster(slug, name, [CharacterEntry("安欣"), CharacterEntry("李响")]) for slug, name in works.items()}
    buffer = io.BytesIO()
    Image.new("RGB", (16, 9), "#6789ab").save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    samples, images, tags, old = {}, {}, [], {}
    for i in range(count):
        sid = f"s{i:04d}"
        slug = list(works)[i % len(works)]
        names = ("安欣", "李响") if i < (47 if count == 160 else 1) else ("安欣",)
        persons = [{"person_id": pid, "annotation_id": f"{sid}-{pid}", "character": name,
                    "bbox": [(pid - 1) / len(names), 0, pid / len(names), 1], "identity_status": "verified", "review_status": "approved", "character_aliases": []}
                   for pid, name in enumerate(names, 1)]
        samples[sid] = {"sample_id": sid, "work_slug": slug, "work_name": works[slug], "image_sha256": "source",
                        "image_width": 16, "image_height": 9,
                        "foreground_complete": True, "review_status": "approved", "persons": persons, "ignore_regions": []}
        images[sid] = {"bytes": image_bytes, "size": [16, 9], "sha256": hashlib.sha256(image_bytes).hexdigest()}
        tags += [{"sample_id": sid, "person_id": p["person_id"], "face_orientation": "front" if p["person_id"] == 1 else "profile", "visibility": "clear"} for p in persons]
        for model in report.MODEL_IDS:
            old[(model, sid)] = raw(names)
    return {"run": {"run_id": "new", "status": "running"}, "baseline": {"run_id": "old", "dataset_hash": "oldhash"},
            "samples": samples, "models": report.MODEL_IDS, "matcher": CharacterMatcher(rosters),
            "tags": tags, "images": images, "old": old, "new": {}, "dataset_hash": "currenthash"}


def review_context(count=2):
    data = context(count)
    top = {"deepseek-flash", "qwen3.8-max", "kimi-k3.1-pro"}
    data["new"] = {key: deepcopy(value) if key[0] in top else raw((None,)) for key, value in data["old"].items()}
    data["run"]["status"] = "done"
    return data


def final_review_context():
    """160 images, 213 required identities, 219 physical people, all API-complete."""
    data = review_context(160)
    for index in range(47, 59):
        sid = f's{index:04d}'
        sample = data['samples'][sid]
        sample['persons'][0]['bbox'] = [0, 0, .5, 1]
        sample['persons'].append({'person_id': 2, 'annotation_id': f'{sid}-2', 'character': '李响',
            'bbox': [.5, 0, 1, 1], 'identity_status': 'verified', 'review_status': 'approved', 'character_aliases': []})
        data['tags'].append({'sample_id': sid, 'person_id': 2, 'face_orientation': 'profile', 'visibility': 'clear'})
        if index < 53:
            for model in report.MODEL_IDS:
                data['old'][(model, sid)] = raw(('安欣', '李响'))
                data['new'][(model, sid)] = raw(('安欣', '李响'))
        else:
            sample['evaluation_policy'] = {'optional_person_ids': [2]}
    return data


def test_brief_main_table_has_three_columns_and_uses_complete_required_name_and_api_denominators():
    data = final_review_context()
    # A failed two-person image must remain in the 213 required-name targets;
    # otherwise the API-success subset would misleadingly show 100% here.
    data['new'][('qwen3.8-max', 's0000')] = raw(('安欣', '李响'), ok=False)
    summary = report.build_summary(data)
    metric = summary['all_works']['qwen3.8-max']['new']
    assert summary['planned_targets_per_model'] == 213 and summary['strict_targets_per_model'] == 219
    assert metric['name_correct'] == 211 and metric['api_ok_targets'] == 211
    assert metric['name_accuracy_api_ok'] == 1
    rows = table_rows(report.brief_results_table(summary))
    assert len(rows) == 8 and all(len(row) == 3 for row in rows)
    assert rows[0][0].text() == '模型'
    assert '姓名命中率' in rows[0][1].text() and 'API调用成功率' in rows[0][2].text()
    by_model = {row[0].text().strip(): row for row in rows[1:]}
    assert set(by_model) == {report.MODEL_LABELS.get(mid, mid) for mid in report.MODEL_IDS}
    selected = by_model[report.MODEL_LABELS['qwen3.8-max']]
    name_text, api_text = selected[1].text(), selected[2].text()
    assert '99.06%' in name_text and re.search(r'211\s*/\s*213', name_text)
    assert not re.search(r'211\s*/\s*211', name_text)
    assert '99.38%' in api_text and re.search(r'159\s*/\s*160', api_text)
    rendered = HtmlTree(report.render_report(summary, data['images'])).root
    main_tables = [node for node in rendered.find_all('table')
                   if len(node.find_all('th')) == 3 and 'API调用成功率' in node.text()]
    assert len(main_tables) == 1


def test_case_uses_one_image_in_left_visual_and_both_model_analyses_on_right():
    data = review_context()
    for model in report.REVIEW_MODELS:
        data['new'][(model, 's0000')] = raw(('安欣',))
    summary = report.build_summary(data)
    html = report.render_report(summary, data['images'])
    dom = HtmlTree(html).root
    cards = dom.find_all('article', 'case-card')
    assert len(cards) == 1 and cards[0].attrs['id'] == 's0000'
    body, = cards[0].find_all(class_name='case-body')
    children = [child for child in body.children if isinstance(child, HtmlNode)]
    visual = body.find_all(class_name='case-visual')[0]
    analysis = body.find_all(class_name='case-analysis')[0]
    assert visual in children and analysis in children and children.index(visual) < children.index(analysis)
    assert len(visual.find_all('img')) == 1 and not analysis.find_all('img')
    assert len(dom.find_all('img')) == 1 and html.count('src="data:image/jpeg;base64,') == 1
    image = visual.find_all('img')[0]
    import base64
    assert hashlib.sha256(base64.b64decode(image.attrs['src'].split(',', 1)[1])).hexdigest() == data['images']['s0000']['sha256']
    assert visual.find_all('button', 'zoom')
    selector, = visual.find_all('select', 'prediction-model')
    assert 'multiple' not in selector.attrs
    assert {option.attrs.get('value') for option in selector.find_all('option')} == {'', *report.REVIEW_MODELS}
    layers = visual.find_all(class_name='prediction-layer')
    assert {node.attrs['data-model'] for node in layers} == set(report.REVIEW_MODELS)
    assert all('hidden' in node.attrs for node in layers)
    panels = analysis.find_all(class_name='model-panel')
    assert {node.attrs['data-model'] for node in panels} == set(report.REVIEW_MODELS)
    for panel in panels:
        original, = panel.find_all('details', 'raw-responses')
        assert 'open' not in original.attrs
        assert '原始回答' in original.find_all('summary')[0].text()
        assert data['new'][(panel.attrs['data-model'], 's0000')]['response']['raw_text'] in original.text()


def test_classification_is_collapsed_and_current_review_hash_is_visible_without_rescoring_or_raw_mutation():
    data = review_context()
    data['new'][('deepseek-flash', 's0000')] = raw(('安欣',))
    before = deepcopy(data)
    original = report.build_summary(data)
    data['dataset_hash'] = 'c' * 64
    data['annotation_review'] = {'applied': True, 'current_dataset_hash': 'c' * 64,
        'source_dataset_hash': 'b' * 64, 'reviewed_sample_ids': ['s0000'], 'reviewed_sample_count': 1,
        'manifest_sha256': 'd' * 64, 'policy': 'Offline rescoring of saved responses; no new model calls.'}
    summary = report.build_summary(data)
    assert summary['all_works'] == original['all_works'] and summary['groups'] == original['groups']
    assert summary['cases'] == original['cases']
    frozen_summary = deepcopy(summary)
    dom = HtmlTree(report.render_report(summary, data['images'])).root
    assert summary == frozen_summary and data['old'] == before['old'] and data['new'] == before['new']
    assert dom.find_all('h1')[0].text() == '联网角色识别测试'
    intro, = dom.find_all('header')
    assert '最终复核' in intro.text()
    version, = intro.find_all(class_name='review-status')
    assert version.attrs['data-dataset-hash'] == summary['dataset_hash']
    assert summary['dataset_hash'] in version.attrs['title']
    classification = [node for node in dom.find_all('details')
                      if node.find_all('summary') and '分类统计' in node.find_all('summary')[0].text()]
    assert len(classification) == 1 and 'open' not in classification[0].attrs
    assert classification[0].find_all('table')  # Content remains available when expanded.


def test_mismatched_applied_review_hash_is_rejected_instead_of_marked_final():
    data = review_context()
    summary = report.build_summary(data)
    summary['annotation_review'] = {'applied': True, 'current_dataset_hash': 'f' * 64}
    assert summary['annotation_review']['current_dataset_hash'] != summary['dataset_hash']
    before = deepcopy(summary)
    with pytest.raises(ValueError, match='review version'):
        report.render_report(summary, data['images'])
    assert summary == before


def test_pending_full_run_retains_1120_tasks_and_207_targets_per_model():
    data = context(160)
    data["new"][("qwen3.8-max", "s0000")] = raw(("安欣", "李响"))
    summary = report.build_summary(data)
    assert summary["planned_tasks"] == 1120
    assert summary["planned_targets_per_model"] == 207
    assert summary["completed_tasks"] == 1 and not summary["complete"]
    assert summary["pending_tasks"] == 1119
    row = summary["all_works"]["qwen3.8-max"]["new"]
    assert row["planned_images"] == 160 and row["planned_targets"] == 207
    assert row["name_correct"] == 2 and row["name_accuracy_planned"] == pytest.approx(2 / 207)
    assert row["name_accuracy_completed"] == 1
    assert row["format_rate_completed"] == 1
    assert row["pending_images"] == 159 and row["format_error_images"] == 0


def test_invalid_coordinates_keep_correct_names_but_fail_strict_scoring():
    data = context()
    record = raw(("安欣", "李响"), invalid_box=True)
    row = report.score_record(data["samples"]["s0000"], record, data["matcher"], {})
    assert row["names"]["correct"] == 2
    assert row["strict"]["identity_correct"] == 0
    assert not row["strict"]["parse_ok"] and row["strict"]["format_error"]
    html = report.model_panel("qwen", {"old": row, "new": row})
    assert "坐标超出0–1范围" in html and "安欣" in html


def test_api_failure_never_credits_an_answer_like_string():
    data = context()
    row = report.score_record(data["samples"]["s0000"], raw(("安欣", "李响"), ok=False), data["matcher"], {})
    assert row["completed"] and row["state"] == "api_failure"
    assert row["names"]["correct"] == 0 and row["strict"]["identity_correct"] == 0


def test_only_valid_retrieved_source_urls_prove_search():
    metadata = {"requested": True, "attempted": True, "retrieved": True,
                "sources": [{"url": "javascript:alert(1)"}, {"url": "https://["}, {"url": "https://user:secret@example.com"}],
                "billing": {"known_units": False, "units": {"search": 1}}, "queries": ["作品角色", "作品角色"]}
    missing = report.evidence(raw(metadata=metadata))
    assert missing["requested"] and missing["attempted"]
    assert not missing["retrieved"] and not missing["sources"]
    metadata["sources"] += [{"url": "https://example.com/role", "title": "真实来源"}, {"url": "https://example.com/role"}]
    result = report.evidence(raw(metadata=metadata))
    assert result["retrieved"] and len(result["sources"]) == 1
    assert result["queries"] == ["作品角色"] and not result["billing"]["known_units"]
    metadata["sources"] = None
    assert not report.evidence(raw(metadata=metadata))["retrieved"]


def test_native_tool_intent_and_http200_tool_errors_are_not_sources():
    record = raw()
    record["response"]["usage"] = {"server_tool_use": {"web_search_requests": 3}}
    record["response"]["search_metadata"] = {"messages": [{"search_info": {"web_search_tool_results": [{"content": {"type": "web_search_tool_result_error", "error_code": "failure"}}]}}]}
    result = report.evidence(record)
    assert result["attempted"] is True and not result["retrieved"]
    assert not result["billing"]["known_units"]


def test_grouping_uses_gt_person_pose_and_keeps_excluded_work_separate():
    data = context()
    data["new"][("qwen3.8-max", "s0000")] = raw(("安欣",))
    summary = report.build_summary(data)
    front = summary["groups"]["orientation"]["front"]["qwen3.8-max"]["new"]
    profile = summary["groups"]["orientation"]["profile"]["qwen3.8-max"]["new"]
    assert front["planned_targets"] == 2 and front["name_correct"] == 1
    assert profile["planned_targets"] == 1 and profile["name_correct"] == 0
    assert summary["groups"]["single_multi"]["multi"]["qwen3.8-max"]["new"]["planned_images"] == 1
    assert summary["without_gongfunvzu"]["qwen3.8-max"]["new"]["planned_targets"] == 2
    assert summary["all_works"]["qwen3.8-max"]["new"]["planned_targets"] == 3


def test_empty_classification_has_no_ranked_percentage_but_keeps_all_models_available():
    data = context()
    summary = report.build_summary(data)
    empty = summary['groups']['orientation']['not_visible']
    assert all(row['new']['planned_targets'] == 0 and row['new']['name_accuracy_planned'] is None
               for row in empty.values())
    rows = table_rows(report.comparison_table(empty))
    assert len(rows) == 8
    assert {row[0].text() for row in rows[1:]} == {report.MODEL_LABELS.get(mid, mid) for mid in report.MODEL_IDS}
    assert all('%' not in cell.text() for row in rows[1:] for cell in row[1:])
    assert all(row[1].text() == row[2].text() == '—' for row in rows[1:])


def test_truth_validation_accepts_evidence_prose_but_rejects_geometry_or_names():
    samples = context()["samples"]
    changed = deepcopy(samples)
    changed["s0000"]["note"] = "用户更正说明"
    changed["s0000"]["persons"][0]["evidence"] = [{"observation": "新的说明"}]
    assert report.scoring_truth(changed) == report.scoring_truth(samples)
    changed["s0000"]["persons"][0]["character"] = "李响"
    assert report.scoring_truth(changed) != report.scoring_truth(samples)
    changed = deepcopy(samples)
    changed["s0000"]["persons"][0]["bbox"] = [.1, .1, .9, .9]
    assert report.scoring_truth(changed) != report.scoring_truth(samples)


def test_html_embeds_one_input_per_sample_and_escapes_response_content():
    data = review_context()
    dangerous = raw(("</script><script>alert('x')</script>",))
    dangerous["response"]["search_metadata"] = {"requested": True, "attempted": True, "retrieved": True,
        "sources": [{"url": "https://example.com/role", "title": "<img onerror=alert(1)>"}]}
    data["new"][("qwen3.8-max", "s0000")] = dangerous
    html = report.render_report(report.build_summary(data), data["images"])
    assert html.count('src="data:image/jpeg;base64,') == 1  # The other input is correct for all Top3 and is not embedded.
    assert html.count('class="model-panel"') == 1
    assert html.count('<script>') == 1
    assert "<img onerror=" not in html and "&lt;img onerror=" in html
    assert 'src="http' not in html and 'href="https://example.com/role"' in html
    cards = HtmlTree(html).root.find_all('article', 'case-card')
    assert len(cards) == 1 and cards[0].attrs['id'] == 's0000'


def test_export_writes_matching_html_hash_without_model_calls(tmp_path, monkeypatch):
    data = context()
    monkeypatch.setattr(report, "read_context", lambda *_args: data)
    summary = report.export_report("new", tmp_path)
    assert (tmp_path / "index.html").exists()
    stored = json.loads((tmp_path / "summary.json").read_text())
    assert stored["html_sha256"] == hashlib.sha256((tmp_path / "index.html").read_bytes()).hexdigest()
    assert summary["planned_tasks"] == 14 and summary["completed_tasks"] == 0


def test_methods_and_budget_remain_auditable_outside_the_brief_main_interface():
    data = context()
    data["metadata"] = {"budget_scope": {"max_cost_cny": 350}, "per_model_notes": {
        "glm-4.6v": {"request_params": {"api_mode": "zhipu_function_search", "max_client_search_calls": 8, "extra_body": {"thinking": {"type": "disabled"}}}},
        "qwen3.8-max": {"request_params": {"api_mode": "dashscope_anthropic", "search_max_uses": 8, "extra_body": {"enable_thinking": False}}}}}
    summary = report.build_summary(data)
    methods = report.methods_table(summary)
    assert summary['configuration']['budget_scope']['max_cost_cny'] == 350
    assert "智谱自有搜索API＋模型function工具循环" in methods
    assert "声明8次，仅为建议，服务端可能超出" in methods
    assert "客户端每图最多8次搜索" in methods
    comparison = report.comparison_table(summary['all_works'])
    assert "待全部完成" in comparison and "-100.00 个百分点" not in comparison


def test_request_image_redaction_preserves_actual_prompts_hashes_and_original_evidence():
    image = context()["images"]["s0000"]["bytes"]
    import base64
    encoded = base64.b64encode(image).decode()
    request = {"system": "原系统提示词", "user": "原业务提示词", "image_meta": {"encoded_sha256": hashlib.sha256(image).hexdigest()},
               "messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
                                          {"type": "text", "text": "原始文字"},
                                          {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}}]}]}
    before = deepcopy(request)
    projected = report.report_request(request)
    assert request == before
    assert projected["system"] == request["system"] and projected["user"] == request["user"]
    first = projected["messages"][0]["content"][0]["image_url"]["url"]
    second = projected["messages"][0]["content"][2]["source"]["data"]
    assert first["encoded_image_sha256"] == second["encoded_image_sha256"] == hashlib.sha256(image).hexdigest()
    assert first["byte_length"] == len(image)
    assert encoded not in json.dumps(projected)


def test_native_multimodal_method_and_stopped_run_metadata_remain_available():
    data = context()
    data["run"].update(status="limited", note="Provider violated a token/cost bound")
    data["metadata"] = {"per_model_notes": {"qwen3.7-plus": {"request_params": {
        "api_mode": "dashscope_multimodal_search", "search_max_uses": 8, "extra_body": {"enable_thinking": False}}}}}
    summary = report.build_summary(data)
    summary["stage_label"] = "首轮56项接入验证（已停止）"
    methods = report.methods_table(summary)
    assert "Qwen原厂DashScope多模态流式搜索" in methods
    assert "声明8次，仅为建议" not in methods
    assert summary['stage_label'] == '首轮56项接入验证（已停止）'
    assert summary['status'] == 'limited' and not summary['complete']


def test_accounting_output_allowance_does_not_claim_changed_wire_generation():
    data = context()
    data["metadata"] = {"generation_params": {"max_tokens": 4096}, "pricing": {"models": {
        "qwen3.8-max": {"output_token_upper_bound": 131072}}}}
    original = deepcopy(data['metadata'])
    summary = report.build_summary(data)
    assert summary['configuration']['generation_params']['max_tokens'] == 4096
    assert summary['configuration']['pricing']['models']['qwen3.8-max']['output_token_upper_bound'] == 131072
    assert data['metadata'] == original


def test_explicit_error_evidence_separates_account_filter_and_unknown_without_mutation():
    data = review_context()
    account = raw(("安欣", "李响"), ok=False)
    account["response"].update(error_type="bad_request", error_message="HTTP400 {'code':'Arrearage'}")
    content = raw(ok=False)
    content["response"].update(error_type="bad_request", error_message="系统检测到输入或生成内容可能包含不安全或敏感内容")
    unknown = raw(ok=False)
    unknown["response"].update(error_type="bad_request", error_message="Provider returned a non-JSON HTTP error")
    originals = deepcopy((account, content, unknown))
    assert report.failure_category(account["response"]) == "account_arrearage"
    assert report.failure_category(content["response"]) == "content_filter"
    assert report.failure_category({"ok": False, "error_type": "bad_request", "error_message": "HTTP400 {'code':'DataInspectionFailed','message':'Input text data may contain inappropriate content.'}"}) == "content_filter"
    assert report.failure_category(unknown["response"]) == "other_api_failure"
    data["new"][("qwen3.8-max", "s0000")] = account
    data["new"][("qwen3.7-plus", "s0000")] = unknown
    summary = report.build_summary(data)
    account_stats = summary["all_works"]["qwen3.8-max"]["new"]
    assert account_stats["api_failure_categories"]["account_arrearage"] == 1
    assert account_stats["api_ok_targets"] == 1 and account_stats["name_accuracy_api_ok"] == 1
    assert summary["all_works"]["qwen3.7-plus"]["new"]["api_failure_categories"]["other_api_failure"] == 1
    html = report.render_report(summary, data["images"])
    assert "接口失败，未判定姓名" in html
    assert "其他 / 原因未确认" in report.failures_table(summary)
    assert (account, content, unknown) == originals


def test_local_reserve_ledger_and_paused_state_are_preserved_without_bloating_the_main_report():
    data = context()
    data["run"].update(status="limited", note="Rolling budget settled")
    data["costs"] = {"cumulative_scope": {"committed_cny": 343.8035597, "known_token_estimate_cny": 72.7983437,
                      "uncertain_cny": 271.005216, "inflight_reserve_cny": 0}}
    summary = report.build_summary(data)
    summary["pause_note"] = "已暂停，等待本地累计预算指示；充值不等于自动提高本地上限。"
    assert summary['status'] == 'limited' and not summary['complete']
    assert summary['costs'] == data['costs']
    ledger = report.ledger_totals([
        {'known_cny': 2, 'uncertain_cny': 3, 'committed_cny': 5, 'status': 'settled'},
        {'known_cny': 0, 'uncertain_cny': 0, 'committed_cny': 7, 'status': 'in_flight'},
    ])
    assert ledger == {'known_token_estimate_cny': 2, 'uncertain_cny': 3,
                      'inflight_reserve_cny': 7, 'committed_cny': 12, 'http_attempt_rows': 2}
    html = report.render_report(summary, data['images'])
    assert '¥343.803560' not in html and '¥271.005216' not in html


def test_provider_budget_hold_does_not_claim_entire_running_test_is_paused():
    data = context()
    data["run"].update(status="running", note="此前中断说明")
    data["operational_control"] = {"provider_holds": {"dashscope": {"state": "held", "kind": "local_budget_allocation"}},
                                   "balance_confirmation_providers": ["dashscope"]}
    summary = report.build_summary(data)
    html = report.render_report(summary, data["images"])
    assert summary['operational_control'] == data['operational_control']
    assert summary['status'] == 'running' and not summary['complete']
    assert "阿里云Qwen当前调用控制" not in html
    assert "运行记录：此前中断说明" not in html


def composite_context(*, flash_complete=False, flash_status="running"):
    data = context()
    data["run"].update(run_id="primary", status="limited")
    original = deepcopy(data["old"])
    replacement = {(model, sid): value for (model, sid), value in original.items() if model == "qwen3.8-flash"}
    if not flash_complete:
        replacement = {("qwen3.8-flash", "s0000"): raw(("无关角色",))}
    source_ids = {model: "flash_auto" if model == "qwen3.8-flash" else "primary" for model in report.MODEL_IDS}
    data["new"] = report.select_model_records({"primary": original, "flash_auto": replacement}, source_ids)
    data["model_sources"] = {model: {"run_id": rid, "status": flash_status if rid == "flash_auto" else "limited",
                                     "selection_reason": "接口稳定性和来源预检，非逐图择优"} for model, rid in source_ids.items()}
    data["diagnostics"] = {"qwen3.8-flash": {"run_id": "primary", "replaced_by_run_id": "flash_auto",
        "records": {sid: value for (model, sid), value in original.items() if model == "qwen3.8-flash"},
        "request_params": {"extra_body": {"enable_thinking": False, "search_options": {"forced_search": True}}},
        "costs": {"known_token_estimate_cny": .1, "uncertain_cny": .2, "committed_cny": .3}}}
    return data


def test_whole_model_override_never_fills_pending_images_or_picks_the_better_old_answer():
    data = composite_context()
    assert ("qwen3.8-flash", "s0001") not in data["new"]
    summary = report.build_summary(data)
    flash = summary["all_works"]["qwen3.8-flash"]["new"]
    assert flash["completed_images"] == 1 and flash["pending_images"] == 1
    assert flash["name_correct"] == 0  # Old cohort recognized all three; none is copied.
    assert summary["completed_tasks"] == 13 and summary["pending_tasks"] == 1
    assert summary["status"] == "running" and summary["primary_run_status"] == "limited"
    assert summary["model_sources"]["qwen3.8-flash"]["run_id"] == "flash_auto"
    assert all(case["results"]["qwen3.8-flash"]["new"]["source_run_id"] == "flash_auto" for case in summary["cases"])
    diagnostic = summary["excluded_model_diagnostics"]["qwen3.8-flash"]
    assert diagnostic["recorded_images"] == 2 and diagnostic["metrics_recorded_subset"]["name_correct"] == 3
    assert diagnostic["costs"]["committed_cny"] == .3


def test_composite_completes_with_limited_primary_and_finished_whole_model_replacement():
    data = composite_context(flash_complete=True, flash_status="done")
    summary = report.build_summary(data)
    assert summary["complete"] and summary["finalized"] and summary["status"] == "done"
    assert summary["primary_run_status"] == "limited" and summary["pending_tasks"] == 0
    assert summary["completed_tasks"] == 14
    data["model_sources"]["qwen3.8-flash"]["status"] = "running"
    finishing = report.build_summary(data)
    assert finishing["complete"] and not finishing["finalized"] and finishing["status"] == "running"
    assert "仍在收尾或重试" in report.render_report(finishing, data["images"])


def test_full_160_composite_counts_1120_selected_tasks_and_excludes_all_old_flash_rows():
    data = context(160)
    data["run"].update(run_id="primary", status="limited")
    sources = {model: "fresh_flash" if model == "qwen3.8-flash" else "primary" for model in report.MODEL_IDS}
    data["model_sources"] = {model: {"run_id": rid, "status": "running" if rid == "fresh_flash" else "limited"}
                             for model, rid in sources.items()}
    per_run = {"primary": data["old"], "fresh_flash": {}}
    data["new"] = report.select_model_records(per_run, sources)
    pending = report.build_summary(data)
    assert pending["planned_tasks"] == 1120 and pending["planned_targets_per_model"] == 207
    assert pending["completed_tasks"] == 960 and pending["pending_tasks"] == 160
    assert pending["all_works"]["qwen3.8-flash"]["new"]["completed_images"] == 0
    per_run["fresh_flash"] = {(model, sid): value for (model, sid), value in data["old"].items() if model == "qwen3.8-flash"}
    data["new"] = report.select_model_records(per_run, sources)
    data["model_sources"]["qwen3.8-flash"]["status"] = "done"
    completed = report.build_summary(data)
    assert completed["completed_tasks"] == 1120 and completed["pending_tasks"] == 0
    assert completed["complete"] and completed["finalized"] and completed["status"] == "done"
    assert completed["primary_run_status"] == "limited"


def test_override_manifest_and_explicit_cli_selection_are_model_level_and_primary_bound(tmp_path):
    path = tmp_path / report.OVERRIDES_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"primary_run_id": "primary", "model_runs": {"qwen3.8-flash": "auto"}, "selection_reason": "stability/source proof"}))
    assert report.resolve_model_runs(tmp_path, "primary")[0] == {"qwen3.8-flash": "auto"}
    assert report.resolve_model_runs(tmp_path, "primary", {"qwen3.8-flash": "max"})[0] == {"qwen3.8-flash": "max"}
    assert report.resolve_model_runs(tmp_path, "unrelated")[0] == {}
    assert report.resolve_model_runs(tmp_path, "primary", {"qwen3.8-flash": "primary"})[0] == {}
    assert report.parse_model_runs(["qwen3.8-flash=auto", "qwen3.8-max=max"]) == {"qwen3.8-flash": "auto", "qwen3.8-max": "max"}
    for values in (["qwen3.8-flash=auto", "qwen3.8-flash=max"], ["unknown=auto"], ["qwen3.8-flash="]):
        with pytest.raises(ValueError):
            report.parse_model_runs(values)


def profile_metadata():
    return {"benchmark_protocol": "multi-v2", "is_mock": False,
            "budget_scope": {"root_run_id": "shared"}, "generation_params": {"temperature": .1, "max_tokens": 4096},
            "per_model_notes": {"qwen3.8-flash": {"passes_temperature": True, "token_limit_parameter": "max_tokens",
                "request_params": {"api_mode": "dashscope_anthropic", "extra_body": {"enable_thinking": False,
                    "enable_search": True, "search_options": {"forced_search": True}}}}}, "settings_snapshot": {}}


def test_source_validation_allows_native_auto_or_sse_but_rejects_model_scope_generation_or_thinking_changes():
    original = profile_metadata()
    source = {"models": json.dumps(["qwen3.8-flash"]), "experiments": '["C"]'}
    auto = deepcopy(original)
    auto["per_model_notes"]["qwen3.8-flash"]["request_params"]["extra_body"]["search_options"]["forced_search"] = False
    report.validate_source_metadata("qwen3.8-flash", source, auto, original)
    sse = deepcopy(auto)
    sse["per_model_notes"]["qwen3.8-flash"]["request_params"]["api_mode"] = "dashscope_multimodal_search"
    report.validate_source_metadata("qwen3.8-flash", source, sse, original)
    for key, replacement in (("models", '["qwen3.8-max"]'), ("experiments", '["B"]')):
        changed = {**source, key: replacement}
        with pytest.raises(ValueError):
            report.validate_source_metadata("qwen3.8-flash", changed, auto, original)
    for mutation in (lambda d: d.update(is_mock=True),
                     lambda d: d["budget_scope"].update(root_run_id="different"),
                     lambda d: d["generation_params"].update(max_tokens=8192),
                     lambda d: d["per_model_notes"]["qwen3.8-flash"]["request_params"]["extra_body"].update(enable_thinking=True),
                     lambda d: d["per_model_notes"]["qwen3.8-flash"]["request_params"].update(api_mode="moonshot_responses")):
        changed = deepcopy(auto)
        mutation(changed)
        with pytest.raises(ValueError):
            report.validate_source_metadata("qwen3.8-flash", source, changed, original)


def test_business_input_validation_distinguishes_protocol_marker_from_business_prompt():
    baseline = {"model": "qwen3.8-flash", "user": "原无框C业务提示与作品名", "temperature": .1, "max_tokens": 4096}
    request = {**baseline, "system": report.PROTOCOL_MARKER + "\n影视角色业务系统提示", "extra_body": {"enable_thinking": False}}
    report.validate_business_request("qwen3.8-flash", request, baseline, "影视角色业务系统提示", {"enable_thinking": False})
    request["system"] = "影视角色业务系统提示"  # Native multimodal does not carry the protocol marker.
    report.validate_business_request("qwen3.8-flash", request, baseline, "影视角色业务系统提示", {"enable_thinking": False})
    for key, value in (("user", "改变作品或角色提示"), ("system", "改变业务系统提示"), ("temperature", .6), ("model", "qwen3.8-max"), ("extra_body", {"enable_thinking": True})):
        changed = {**request, key: value}
        with pytest.raises(ValueError):
            report.validate_business_request("qwen3.8-flash", changed, baseline, "影视角色业务系统提示", {"enable_thinking": False})


def test_frozen_source_validation_rejects_truth_sidecar_and_physical_image_changes(tmp_path, monkeypatch):
    data = context()
    image = data["images"]["s0000"]["bytes"]
    image_hash = hashlib.sha256(image).hexdigest()

    class FrozenSample:
        def __init__(self, row):
            self.row = row
            self.sample_id = row["sample_id"]
            self.image_sha256 = row["image_sha256"]
            self.image_path = row["image_path"]

        def to_dict(self):
            return deepcopy(self.row)

    current = []
    for sid, row in data["samples"].items():
        current.append(FrozenSample({**deepcopy(row), "image_sha256": image_hash, "image_path": f"{sid}.jpg"}))
        (tmp_path / f"{sid}.jpg").write_bytes(image)
    frozen = list(current)
    monkeypatch.setattr(report, "load_dataset", lambda _path: frozen)
    monkeypatch.setattr(report, "load_frozen_stratification", lambda *_args: {"persons": data["tags"]})
    source = {"run_id": "candidate", "dataset_path": str(tmp_path / "dataset.json"), "dataset_hash": report.dataset_hash(current)}
    metadata = {"image_hashes": {s.sample_id: image_hash for s in current}, "stratification": {"sha256": "same"}}
    report.validate_frozen_source(source, metadata, current, data["tags"], {"sha256": "same"})
    with pytest.raises(ValueError, match="stratification"):
        report.validate_frozen_source(source, metadata, current, data["tags"], {"sha256": "different"})
    frozen = current[:-1]
    with pytest.raises(ValueError, match="confirmed truth"):
        report.validate_frozen_source(source, metadata, current, data["tags"], {"sha256": "same"})
    frozen = list(current)
    (tmp_path / "s0000.jpg").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="image bytes"):
        report.validate_frozen_source(source, metadata, current, data["tags"], {"sha256": "same"})


def test_composite_export_preserves_excluded_diagnostics_separately_and_embeds_only_one_image_each(tmp_path, monkeypatch):
    data = composite_context()
    monkeypatch.setattr(report, "read_context", lambda *_args: data)
    summary = report.export_report("primary", tmp_path, model_runs={"qwen3.8-flash": "flash_auto"})
    diagnostics = json.loads((tmp_path / "excluded_model_diagnostics.json").read_text())
    assert diagnostics["qwen3.8-flash"]["recorded_images"] == 2
    html = (tmp_path / "index.html").read_text()
    assert html.count('src="data:image/jpeg;base64,') == 0  # In this fixture all ranked Top3 are correct.
    assert '按模型整体选择运行来源' not in html and '历史接口逐图原始回答' not in html
    assert summary["model_sources"]["qwen3.8-flash"]["run_id"] == "flash_auto"


def test_review_embeds_only_error_union_and_omits_correct_model_panels_and_coordinate_only_images():
    data = review_context(4)
    data["new"][("deepseek-flash", "s0000")] = raw(("安欣",))
    data["new"][("kimi-k3.1-pro", "s0000")] = raw(("安欣", "李响", "额外角色"))
    data["new"][("qwen3.8-max", "s0001")] = raw(("安欣",), invalid_box=True)
    data["new"][("qwen3.8-max", "s0002")] = raw(ok=False)
    summary = report.build_summary(data)
    review = summary["review_selection"]
    assert review["top_models"] == ["deepseek-flash", "qwen3.8-max"]
    assert review["selected_image_count"] == 2 and review["error_model_panels"] == 2
    assert {row["sample_id"] for row in review["selected_images"]} == {"s0000", "s0002"}
    assert review["coordinate_only_image_count"] == 1
    assert review["coordinate_only_images_not_embedded"][0]["sample_id"] == "s0001"
    assert "5 required names" in review["ranking_metric"]
    html = report.render_report(summary, data["images"])
    assert html.count('src="data:image/jpeg;base64,') == 2
    assert html.count('class="model-panel"') == 2
    assert 'id="s0001" data-work=' not in html and 'id="s0003" data-work=' not in html
    assert '<option value="both" selected>' in html
    assert html.count('class="review-layer prediction-layer"') == 2
    prediction_layers = re.findall(r'<div class="review-layer prediction-layer"[^>]*>', html)
    assert len(prediction_layers) == 2 and all(" hidden" in layer for layer in prediction_layers)
    assert 'data-model="qwen3.8-max" hidden>' in html
    assert "去掉《功夫女足》" not in html
    assert len(summary["cases"]) == 4 and summary["completed_tasks"] == 28
    assert all(values["new"]["planned_targets"] == 5 for values in summary["all_works"].values())


def test_pending_selected_model_does_not_create_a_coordinate_only_all_models_correct_claim():
    data = review_context(4)
    data["new"][("qwen3.8-max", "s0001")] = raw(("安欣",), invalid_box=True)
    del data["new"][("deepseek-flash", "s0001")]
    summary = report.build_summary(data)
    assert not summary["review_selection"]["coordinate_only_images_not_embedded"]


def test_quota_names_and_full_strict_and_pose_denominators_are_independent():
    names = ["安欣", "李响", "高启强", "孟钰"]
    matcher = CharacterMatcher({"kuangbiao": Roster("kuangbiao", "狂飙", [CharacterEntry(name) for name in names])})
    people = [{"person_id": i + 1, "annotation_id": f"p{i+1}", "character": name, "character_aliases": [],
               "identity_status": "verified", "review_status": "approved", "bbox": [i / 4, 0, (i + 1) / 4, 1]} for i, name in enumerate(names)]
    sample = {"sample_id": "quota", "work_slug": "kuangbiao", "persons": people, "foreground_complete": True,
              "ignore_regions": [], "evaluation_policy": {"required_name_count": 3}}
    response = raw(names[:3])
    response["response"]["raw_text"] = json.dumps({"persons": [{"character": person["character"], "bbox": person["bbox"]} for person in people[:3]]}, ensure_ascii=False)
    tags = {i + 1: {"face_orientation": "front" if i < 2 else "profile", "visibility": "clear"} for i in range(4)}
    row = report.score_record(sample, response, matcher, tags)
    whole = report.aggregate([row])
    assert whole["planned_targets"] == 3 and whole["name_correct"] == 3
    assert whole["strict_planned_targets"] == 4 and whole["strict_correct"] == 3
    assert whole["name_accuracy_planned"] == 1 and whole["strict_accuracy_planned"] == .75
    profile = report.aggregate([row], lambda tag: tag["face_orientation"] == "profile")
    assert profile["planned_targets"] == 2 and profile["name_correct"] == 1
    assert profile["name_denominator_policy"] == "full_verified_person_coverage"
    assert report.coordinate_only_issue(row)
    sample["evaluation_policy"] = {"optional_person_ids": [4]}
    allowed = raw(names)
    optional = report.score_record(sample, allowed, matcher, tags)
    aggregate = report.aggregate([optional])
    assert aggregate["planned_targets"] == 3 and aggregate["name_correct"] == 3
    assert aggregate["extra_names"] == 0 and aggregate["name_precision"] == 1
    assert "可选名已命中" in report.model_panel("qwen3.8-max", {"old": optional, "new": optional}, compact=True)


def test_consensus_only_compares_requested_pair_and_excludes_reviewed_from_audit_not_main_errors(tmp_path, monkeypatch):
    from scripts import export_consensus_review as consensus
    data = review_context(4)
    for model in report.REVIEW_MODELS:
        data["new"][(model, "s0000")] = raw(("同答错名",))
        data["new"][(model, "s0001")] = raw(("另一错名",))
    data["annotation_review"] = {"applied": True, "reviewed_sample_ids": ["s0000"]}
    monkeypatch.setattr(report, "read_context", lambda *_args: data)
    summary = report.export_report("new", tmp_path)
    assert {row["sample_id"] for row in summary["review_selection"]["selected_images"]} == {"s0000", "s0001"}
    result = consensus.export_consensus(tmp_path)
    assert result["models"] == ["deepseek-flash", "qwen3.8-max"]
    assert result["two_or_more_model_count"] == 1 and result["items"][0]["sample_id"] == "s0001"
    assert result["excluded_reviewed_sample_ids"] == ["s0000"]
    html = (tmp_path / "consensus_review.html").read_text()
    assert "Kimi" not in html and "原先仅标" not in html
    assert html.count('src="data:image/jpeg;base64,') == 1
