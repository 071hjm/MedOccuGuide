from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from local_qwen_advice import LocalQwenAdviceGenerator
from occupational_health_runtime import (
    RuntimeConfig,
    build_report_text,
    get_runtime,
    render_conclusion_html,
    render_evidence_box_html,
    render_logic_box_html,
    render_process_html,
    render_suggestion_html,
)
from platform_storage import PlatformStorage


APP_TITLE = "职业健康混合智能诊断与受控生成一体化平台"
INPUT_MODE_DATASET = "数据集测试样本"
INPUT_MODE_MANUAL = "手动录入"
INPUT_MODE_UPLOAD = "文件上传"

SAMPLE_HEADERS = ["ID", "时间", "用户", "输入方式", "样本ID", "主检结论", "来源", "建议模式"]
CASE_HEADERS = ["ID", "时间", "用户", "主检结论", "诊断提示", "标注规则命中", "规则建议版本", "备注"]
LOG_HEADERS = ["ID", "时间", "用户", "操作", "详情"]
FINAL_LABEL_CHOICES = ["", "职业禁忌证", "疑似职业病", "复查", "其他疾病或异常", "目前未见异常"]
SOURCE_CHOICES = ["", "expert_system", "lightgbm", "heuristic_fallback"]
RULE_HIT_CHOICES = ["", "命中", "未命中"]

_PREWARM_LOCK = threading.Lock()
_PREWARM_STATE: dict[str, Any] = {
    "started": False,
    "done": False,
    "ok": False,
    "message": "未开始预热",
}


@lru_cache(maxsize=1)
def get_storage() -> PlatformStorage:
    return PlatformStorage()


@lru_cache(maxsize=1)
def get_local_advice_generator() -> LocalQwenAdviceGenerator:
    return get_runtime()._local_advice_generator


def get_prewarm_message() -> str:
    return str(_PREWARM_STATE.get("message") or "未开始预热")


def _background_prewarm() -> None:
    try:
        generator = get_local_advice_generator()
        generator._load()
        _PREWARM_STATE.update({"done": True, "ok": True, "message": "Qwen+LoRA 已完成预热"})
    except Exception as exc:
        _PREWARM_STATE.update({"done": True, "ok": False, "message": f"预热失败：{exc}"})


def start_background_prewarm() -> None:
    with _PREWARM_LOCK:
        if _PREWARM_STATE["started"]:
            return
        _PREWARM_STATE.update({"started": True, "done": False, "ok": False, "message": "Qwen+LoRA 预热中..."})
        threading.Thread(target=_background_prewarm, name="qwen_lora_prewarm", daemon=True).start()


def render_status(text: str, level: str = "info") -> str:
    palette = {
        "info": ("#edf4ff", "#9bb8e8", "#375a95"),
        "success": ("#eef9f0", "#9cd3aa", "#2f6b41"),
        "warning": ("#fff5e8", "#efc485", "#8a5b30"),
        "error": ("#fff0ef", "#e1a4a0", "#8c3e3a"),
    }
    bg, border, color = palette.get(level, palette["info"])
    return (
        f"<div style='padding:8px 10px;border:1px solid {border};"
        f"border-radius:10px;background:{bg};color:{color};font-size:12px;'>{text}</div>"
    )


def render_user_badge(username: str) -> str:
    return (
        "<div style='padding:6px 10px;border:1px solid #bfd0ee;border-radius:999px;"
        f"background:#f1f6ff;color:#46638e;font-size:12px;display:inline-block;'>当前用户：{username}</div>"
        if username
        else ""
    )


def to_json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_user_badge(username: str) -> str:
    return f"<div class='user-badge'>\u5f53\u524d\u7528\u6237\uff1a{username}</div>" if username else ""


def parse_key_value_text(text: str) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "：" in line:
            key, value = line.split("：", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if key:
            record[key] = value
    return record


def parse_record_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    return parse_key_value_text(text)


def parse_file_record(file_path: str | None) -> dict[str, Any]:
    if not file_path:
        return {}
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    return parse_record_text(text)


def toggle_inputs(input_mode: str):
    return (
        gr.update(visible=input_mode == INPUT_MODE_DATASET),
        gr.update(visible=input_mode == INPUT_MODE_MANUAL),
        gr.update(visible=input_mode == INPUT_MODE_UPLOAD),
    )


def build_system_info_html() -> str:
    runtime = get_runtime()
    llm_ok, llm_message = get_local_advice_generator().status()
    prewarm_message = get_prewarm_message()
    return (
        "<div class='summary-section'>"
        "<div class='summary-section-title'>系统信息</div>"
        f"<div class='summary-grid'><div><span>Embedding</span>{runtime.config.embedding_model_name}</div>"
        f"<div><span>向量引擎</span>FAISS</div>"
        f"<div><span>Dense Top-K</span>{runtime.config.dense_top_k}</div>"
        f"<div><span>Sparse Top-K</span>{runtime.config.sparse_top_k}</div>"
        f"<div><span>基础模型</span>{runtime.config.qwen_base_model_dir}</div>"
        f"<div><span>LoRA 目录</span>{runtime.config.qwen_lora_dir}</div>"
        f"<div><span>LoRA 状态</span>{'可用' if llm_ok else '保底模式'}</div>"
        f"<div><span>预热状态</span>{prewarm_message}</div></div>"
        f"<div class='summary-note'>{llm_message}</div>"
        "</div>"
    )


def get_management_tables(
    sample_keyword: str = "",
    sample_label: str = "",
    sample_source: str = "",
    case_keyword: str = "",
    case_label: str = "",
    case_rule_hit: str = "",
    case_rule_version: str = "",
    log_keyword: str = "",
    log_action: str = "",
    log_username: str = "",
):
    storage = get_storage()
    sample_rows = storage.list_sample_records(
        keyword=sample_keyword,
        final_label=sample_label,
        final_source=sample_source,
    )
    case_rows = storage.list_expert_cases(
        keyword=case_keyword,
        final_label=case_label,
        rule_hit_tag=case_rule_hit,
        rule_version=case_rule_version,
    )
    log_rows = storage.list_operation_logs(
        keyword=log_keyword,
        action=log_action,
        username=log_username,
    )
    return sample_rows, case_rows, log_rows


def refresh_management_views(
    sample_keyword: str = "",
    sample_label: str = "",
    sample_source: str = "",
    case_keyword: str = "",
    case_label: str = "",
    case_rule_hit: str = "",
    case_rule_version: str = "",
    log_keyword: str = "",
    log_action: str = "",
    log_username: str = "",
):
    sample_rows, case_rows, log_rows = get_management_tables(
        sample_keyword,
        sample_label,
        sample_source,
        case_keyword,
        case_label,
        case_rule_hit,
        case_rule_version,
        log_keyword,
        log_action,
        log_username,
    )
    return sample_rows, case_rows, log_rows, build_system_info_html()


def empty_outputs(message: str):
    return (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        None,
        None,
        "{}",
        render_status(message, "warning"),
    )


def load_sample_record(sample_id: str):
    runtime = get_runtime()
    sample = runtime.get_sample_by_id(sample_id)
    if not sample:
        return "", "{}", render_status("未找到对应测试样本。", "error")
    record = sample["record"]
    return to_json_text(record), to_json_text(record), render_status(f"已加载数据集样本：{sample['display_name']}", "info")


def write_download_file(prefix: str, text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", prefix=prefix, delete=False)
    with handle:
        handle.write(text)
    return handle.name


def render_probability_html(result: dict[str, Any]) -> str:
    final_result = result.get("final_result") or {}
    probabilities = final_result.get("probabilities") or {}
    source = str(final_result.get("source") or "")
    if not probabilities:
        message = "当前结论来自专家系统直达规则，未经过 LightGBM 三分类概率分布计算。"
        if source == "heuristic_fallback":
            message = "当前使用的是保底启发式判定，未生成稳定的 LightGBM 概率分布。"
        return (
            "<div class='probability-card'>"
            "<div class='probability-title'>LightGBM 三分类概率</div>"
            f"<div class='probability-empty'>{message}</div>"
            "</div>"
        )

    rows: list[str] = []
    sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for label, value in sorted_items:
        prob = float(value)
        width = 0.0 if prob <= 0 else max(2.0, prob * 100.0)
        if prob < 1.0 and width > 98.0:
            width = 98.0
        if 0.99995 <= prob < 1.0:
            display_value = ">99.99%"
        elif 0.0 < prob <= 0.00005:
            display_value = "<0.01%"
        else:
            display_value = f"{prob:.2%}"
        rows.append(
            "<div class='prob-row'>"
            f"<div class='prob-label'>{label}</div>"
            "<div class='prob-track'>"
            f"<div class='prob-bar' style='width:{width:.2f}%;'></div>"
            "</div>"
            f"<div class='prob-value'>{display_value}</div>"
            "</div>"
        )
    return (
        "<div class='probability-card'>"
        "<div class='probability-title'>LightGBM 三分类概率</div>"
        f"{''.join(rows)}"
        "</div>"
    )


def run_local_advice_in_subprocess(result: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    runtime = get_runtime()
    payload = {
        "record": result["record"],
        "final_result": result["final_result"],
        "rag_result": result["rag_result"],
        "query": result["query"],
        "base_model_dir": str(runtime.config.qwen_base_model_dir),
        "adapter_dir": str(runtime.config.qwen_lora_dir),
        "offload_dir": str(runtime.config.qwen_offload_dir),
        "temperature": runtime.config.temperature,
        "top_p": runtime.config.top_p,
        "max_new_tokens": min(int(runtime.config.max_new_tokens), 384),
    }
    with tempfile.TemporaryDirectory(prefix="qwen_advice_") as tmp_dir:
        payload_path = Path(tmp_dir) / "payload.json"
        output_path = Path(tmp_dir) / "result.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        script = r"""
import json
import sys
from pathlib import Path

from local_qwen_advice import LocalQwenAdviceGenerator, LocalQwenConfig

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generator = LocalQwenAdviceGenerator(
    LocalQwenConfig(
        base_model_dir=Path(payload["base_model_dir"]),
        adapter_dir=Path(payload["adapter_dir"]),
        offload_dir=Path(payload["offload_dir"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        max_new_tokens=int(payload["max_new_tokens"]),
    )
)
advice, error = generator.safe_generate(
    payload["record"],
    payload["final_result"],
    payload["rag_result"],
    payload["query"],
)
Path(sys.argv[2]).write_text(json.dumps({"advice": advice, "error": error}, ensure_ascii=False), encoding="utf-8")
"""
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        command = [sys.executable, "-c", script, str(payload_path), str(output_path)]
        completed = subprocess.run(command, cwd=str(Path(r"D:\codex")), env=env, timeout=420, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            return None, stderr or f"子进程异常退出，返回码 {completed.returncode}"
        if not output_path.exists():
            return None, "子进程未写出建议结果。"
        response = json.loads(output_path.read_text(encoding="utf-8"))
        return response.get("advice"), response.get("error")


def enhance_advice_with_local_llm(result: dict[str, Any]) -> dict[str, Any]:
    generator = get_local_advice_generator()
    advice, error = generator.safe_generate(
        result["record"],
        result["final_result"],
        result["rag_result"],
        result["query"],
    )
    if advice is None:
        get_local_advice_generator.cache_clear()
        generator = get_local_advice_generator()
        advice, error = generator.safe_generate(
            result["record"],
            result["final_result"],
            result["rag_result"],
            result["query"],
        )
    if advice is None:
        raise RuntimeError(f"本地 Qwen+LoRA 建议生成失败：{error}")
    result["advice"] = advice
    return result


def login_user(username: str, password: str):
    storage = get_storage()
    ok, message = storage.authenticate(username, password)
    if not ok:
        storage.log_operation(username or "anonymous", "登录失败", message)
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            render_status(message, "error"),
            "",
            [],
            [],
            [],
            build_system_info_html(),
        )

    storage.log_operation(username, "登录平台", f"欢迎 {message}")
    sample_rows, case_rows, log_rows = get_management_tables()
    return (
        username,
        gr.update(visible=False),
        gr.update(visible=True),
        render_status(f"登录成功，欢迎 {message}。", "success"),
        render_user_badge(username),
        sample_rows,
        case_rows,
        log_rows,
        build_system_info_html(),
    )


def logout_user(current_user: str):
    username = (current_user or "").strip()
    if username:
        get_storage().log_operation(username, "退出登录", "用户主动退出平台")
    return (
        "",
        gr.update(visible=True),
        gr.update(visible=False),
        render_status("已退出登录。", "info"),
        "",
    )


def _resolve_input_record(input_mode: str, sample_id: str, manual_text: str, uploaded_file: str | None):
    if input_mode == INPUT_MODE_DATASET:
        sample = get_runtime().get_sample_by_id(sample_id)
        if not sample:
            raise ValueError("请选择有效的数据集样本。")
        return sample["record"], sample["sample_id"], f"来源：{sample['display_name']}"
    if input_mode == INPUT_MODE_UPLOAD:
        record = parse_file_record(uploaded_file)
        if not record:
            raise ValueError("上传文件为空或格式无法识别。")
        return record, "", f"来源：文件上传 {Path(uploaded_file).name}"
    record = parse_record_text(manual_text)
    if not record:
        raise ValueError("手动录入内容为空或格式无法识别。")
    return record, "", "来源：手动录入"


def run_inference(current_user: str, input_mode: str, sample_id: str, manual_text: str, uploaded_file: str | None):
    username = (current_user or "").strip()
    if not username:
        return (*empty_outputs("请先登录后再执行分析。"), [], [], [])

    try:
        record, resolved_sample_id, source_note = _resolve_input_record(input_mode, sample_id, manual_text, uploaded_file)
        runtime = get_runtime()
        result = enhance_advice_with_local_llm(runtime.predict(record))
        record_id = get_storage().save_analysis_record(username, input_mode, resolved_sample_id, result)
        get_storage().log_operation(
            username,
            "执行分析",
            f"样本库记录 #{record_id}，主检结论：{result['final_result']['label']}，建议模式：{result['advice']['mode']}",
        )
        sample_rows, case_rows, log_rows = get_management_tables()
        return (
            runtime.summarize_record(record, source_note),
            render_conclusion_html(result),
            render_probability_html(result),
            render_logic_box_html(result),
            render_evidence_box_html(result),
            render_suggestion_html(result),
            render_process_html(result),
            result,
            None,
            to_json_text(record),
            render_status(f"分析完成，结果已写入样本数据库记录 #{record_id}。", "success"),
            sample_rows,
            case_rows,
            log_rows,
        )
    except Exception as exc:
        get_storage().log_operation(username, "执行分析失败", str(exc))
        return (*empty_outputs(f"分析失败：{exc}"), [], [], [])


def save_current_case(current_user: str, result: dict[str, Any] | None, case_note: str):
    username = (current_user or "").strip()
    if not username:
        return "", render_status("请先登录。", "error"), [], []
    if not result:
        return "", render_status("当前没有可保存的分析结果。", "warning"), [], []
    case_id = get_storage().save_expert_case(username, result, case_note)
    get_storage().log_operation(username, "加入专家案例库", f"案例编号 #{case_id}")
    _, case_rows, log_rows = get_management_tables()
    return "", render_status(f"已加入专家案例库，案例编号 #{case_id}。", "success"), case_rows, log_rows


def export_advice(current_user: str, result: dict[str, Any] | None):
    username = (current_user or "").strip()
    if not username:
        return gr.update(visible=False), render_status("请先登录。", "error"), []
    if not result:
        return gr.update(visible=False), render_status("当前没有可导出的建议。", "warning"), []
    file_path = write_download_file("advice_", result["advice"]["text"])
    get_storage().log_operation(username, "导出建议文本", Path(file_path).name)
    _, _, log_rows = get_management_tables()
    return gr.update(value=file_path, visible=True), render_status("建议文本已生成，可直接下载。", "success"), log_rows


def export_report(current_user: str, result: dict[str, Any] | None):
    username = (current_user or "").strip()
    if not username:
        return gr.update(visible=False), render_status("请先登录。", "error"), []
    if not result:
        return gr.update(visible=False), render_status("当前没有可导出的报告。", "warning"), []
    file_path = write_download_file("report_", build_report_text(result))
    get_storage().log_operation(username, "导出完整报告", Path(file_path).name)
    _, _, log_rows = get_management_tables()
    return gr.update(value=file_path, visible=True), render_status("完整报告已生成，可直接下载。", "success"), log_rows


def _selected_row_id(table_data: list[list[Any]], evt: gr.SelectData) -> str:
    row_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if row_index is None or row_index >= len(table_data):
        raise ValueError("未获取到表格行。")
    return str(table_data[row_index][0])


def replay_sample_record(table_data: list[list[Any]], evt: gr.SelectData):
    record_id = _selected_row_id(table_data, evt)
    payload = get_storage().get_sample_record_by_id(record_id)
    if not payload:
        return INPUT_MODE_MANUAL, "", "{}", "", render_status("未找到该条样本记录。", "error"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    note = f"已从样本数据库回填记录 #{record_id}，原主检结论：{payload['final_label']}。可点击“开始分析”重新复盘。"
    record_text = to_json_text(payload["record"])
    return INPUT_MODE_MANUAL, record_text, record_text, get_runtime().summarize_record(payload["record"], note), render_status(note, "info"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def replay_expert_case(table_data: list[list[Any]], evt: gr.SelectData):
    case_id = _selected_row_id(table_data, evt)
    payload = get_storage().get_expert_case_by_id(case_id)
    if not payload:
        return INPUT_MODE_MANUAL, "", "{}", "", render_status("未找到该条专家案例。", "error"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    note = (
        f"已从专家案例库回填案例 #{case_id}，原主检结论：{payload['final_label']}，"
        f"规则命中：{payload['rule_hit_tag']}，规则版本：{payload['rule_version']}。"
    )
    record_text = to_json_text(payload["record"])
    return INPUT_MODE_MANUAL, record_text, record_text, get_runtime().summarize_record(payload["record"], note), render_status(note, "info"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def build_demo():
    start_background_prewarm()
    runtime = get_runtime()
    sample_choices = [(item["display_name"], item["sample_id"]) for item in runtime.build_demo_samples()]
    default_sample_id = sample_choices[0][1] if sample_choices else ""
    default_record = runtime.get_sample_by_id(default_sample_id)["record"] if default_sample_id else {}
    default_record_text = to_json_text(default_record) if default_record else ""
    default_conclusion = "<div class='alert-strip'><span>检测结论：待分析</span><span>判定来源：待运行 | 诊断提示：待生成</span></div>"
    default_probability = "<div class='probability-card'><div class='probability-title'>LightGBM 三分类概率</div><div class='probability-empty'>分析完成后，这里会显示三分类概率柱状图；若专家系统直接命中，则显示说明信息。</div></div>"
    default_logic = "<div class='mini-box'><h3>规则路径 / 模型路径</h3><ul><li>点击“开始分析”后显示命中规则路径或模型判断路径。</li></ul></div>"
    default_evidence = "<div class='mini-box'><h3>异常项与判定依据</h3><ul><li>分析完成后，这里会显示规则命中信息或 SHAP / 贡献值依据。</li></ul></div>"
    default_suggestion = "<div class='suggestions-body'><p>主检建议区</p><p>分析完成后，这里会展示岗位处置、复查转诊、职业防护与提示说明。</p></div>"

    with gr.Blocks(css=Path(r"D:\codex\gradio_ui.css").read_text(encoding="utf-8"), title=APP_TITLE, fill_height=True) as demo:
        current_user = gr.State("")
        current_result = gr.State(None)

        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                f"<div class='app-title-bar'><div class='app-title-text'>{APP_TITLE}</div></div>",
                elem_classes=["app-title-shell"],
            )

            with gr.Column(visible=True, elem_classes=["login-panel"]) as login_panel:
                login_status = gr.HTML(render_status("请输入账号密码进入平台。"))
                with gr.Row():
                    username = gr.Textbox(label="账号", value="admin")
                    password = gr.Textbox(label="密码", value="admin123", type="password")
                login_btn = gr.Button("登录平台", variant="primary", elem_classes=["primary-btn"])

            with gr.Column(visible=False, elem_id="main-shell") as main_panel:
                with gr.Row(elem_classes=["login-row"]):
                    user_badge = gr.HTML(elem_classes=["user-slot"])
                    logout_btn = gr.Button("退出登录", min_width=88, elem_classes=["ghost-btn", "logout-btn"])
                top_status_html = gr.HTML(render_status("请选择样本并开始分析。"))

                with gr.Row(equal_height=True, elem_classes=["content-grid"]):
                    with gr.Column(scale=4, min_width=520, elem_classes=["left-column"]):
                        input_mode = gr.Radio(
                            [INPUT_MODE_DATASET, INPUT_MODE_MANUAL, INPUT_MODE_UPLOAD],
                            value=INPUT_MODE_DATASET,
                            label="样本输入模块",
                        )
                        with gr.Column(visible=True) as dataset_box:
                            sample_id = gr.Dropdown(sample_choices, value=default_sample_id, label="测试样本")
                        with gr.Column(visible=False) as manual_box:
                            manual_text = gr.Textbox(label="手动录入", lines=14, value=default_record_text)
                        with gr.Column(visible=False) as upload_box:
                            uploaded_file = gr.File(label="上传 JSON / 文本", type="filepath")
                        analyze_btn = gr.Button("开始分析", variant="primary", elem_classes=["primary-btn", "hero-btn"])
                        current_json = gr.Textbox(label="当前样本 JSON", lines=12, value=default_record_text, elem_classes=["current-json"])
                        summary_html = gr.HTML(
                            runtime.summarize_record(default_record, "来源：默认数据集样本") if default_record else "",
                            elem_classes=["summary-pane"],
                        )

                    with gr.Column(scale=5, min_width=620, elem_classes=["right-column"]):
                        conclusion_html = gr.HTML(default_conclusion, elem_classes=["result-card", "green-card", "top-result-card"])
                        probability_html = gr.HTML(default_probability, elem_classes=["result-card", "green-card", "probability-result-card"])
                        with gr.Row(equal_height=True, elem_classes=["two-col-boxes"]):
                            logic_html = gr.HTML(default_logic, elem_classes=["result-card", "green-card", "equal-box"])
                            evidence_html = gr.HTML(default_evidence, elem_classes=["result-card", "green-card", "equal-box"])
                        suggestion_html = gr.HTML(default_suggestion, elem_classes=["result-card", "warm-card", "bottom-suggestion-card"])
                        with gr.Row(elem_classes=["export-row"]):
                            export_advice_btn = gr.Button("导出建议文本", elem_classes=["ghost-btn"])
                            export_report_btn = gr.Button("导出完整报告", elem_classes=["ghost-btn"])
                        advice_download = gr.File(label="建议文本下载", visible=False, elem_classes=["download-file"])
                        report_download = gr.File(label="完整报告下载", visible=False, elem_classes=["download-file"])

                with gr.Tabs():
                    with gr.Tab("分析过程"):
                        process_html = gr.HTML()

                    with gr.Tab("样本数据库"):
                        with gr.Row():
                            sample_keyword = gr.Textbox(label="搜索关键字", placeholder="用户 / 样本ID / 结论 / 诊断")
                            sample_label = gr.Dropdown(FINAL_LABEL_CHOICES, value="", label="主检结论筛选")
                            sample_source = gr.Dropdown(SOURCE_CHOICES, value="", label="判定来源筛选")
                            sample_search_btn = gr.Button("搜索样本库")
                        gr.Markdown("点击某条记录可回填到分析界面重新复盘。")
                        sample_table = gr.Dataframe(headers=SAMPLE_HEADERS, datatype=["str"] * len(SAMPLE_HEADERS), interactive=False, row_count=(0, "dynamic"), col_count=(len(SAMPLE_HEADERS), "fixed"))

                    with gr.Tab("专家案例库"):
                        with gr.Row():
                            case_keyword = gr.Textbox(label="搜索关键字", placeholder="结论 / 诊断 / 备注")
                            case_label = gr.Dropdown(FINAL_LABEL_CHOICES, value="", label="主检结论筛选")
                            case_rule_hit = gr.Dropdown(RULE_HIT_CHOICES, value="", label="规则命中筛选")
                            case_rule_version = gr.Textbox(label="规则建议版本", placeholder="例如 expert_rules_v1")
                            case_search_btn = gr.Button("搜索案例库")
                        gr.Markdown("点击某条案例可回填到分析界面；支持额外保存当前分析为专家案例。")
                        case_note = gr.Textbox(label="案例备注", lines=2, placeholder="可记录规则修订建议、专家复核意见等")
                        save_case_btn = gr.Button("加入专家案例库")
                        case_table = gr.Dataframe(headers=CASE_HEADERS, datatype=["str"] * len(CASE_HEADERS), interactive=False, row_count=(0, "dynamic"), col_count=(len(CASE_HEADERS), "fixed"))

                    with gr.Tab("操作日志"):
                        with gr.Row():
                            log_keyword = gr.Textbox(label="搜索关键字", placeholder="详情关键字")
                            log_action = gr.Textbox(label="操作类型", placeholder="例如 执行分析")
                            log_username = gr.Textbox(label="用户", placeholder="例如 admin")
                            log_search_btn = gr.Button("搜索日志")
                        log_table = gr.Dataframe(headers=LOG_HEADERS, datatype=["str"] * len(LOG_HEADERS), interactive=False, row_count=(0, "dynamic"), col_count=(len(LOG_HEADERS), "fixed"))

                    with gr.Tab("系统信息"):
                        system_info_html = gr.HTML(build_system_info_html())

                refresh_btn = gr.Button("刷新管理数据")

        login_btn.click(
            login_user,
            inputs=[username, password],
            outputs=[current_user, login_panel, main_panel, login_status, user_badge, sample_table, case_table, log_table, system_info_html],
        )
        logout_btn.click(
            logout_user,
            inputs=[current_user],
            outputs=[current_user, login_panel, main_panel, login_status, user_badge],
        )
        input_mode.change(toggle_inputs, inputs=[input_mode], outputs=[dataset_box, manual_box, upload_box])
        sample_id.change(load_sample_record, inputs=[sample_id], outputs=[manual_text, current_json, top_status_html])
        analyze_btn.click(
            run_inference,
            inputs=[current_user, input_mode, sample_id, manual_text, uploaded_file],
            outputs=[summary_html, conclusion_html, probability_html, logic_html, evidence_html, suggestion_html, process_html, current_result, advice_download, current_json, top_status_html, sample_table, case_table, log_table],
            show_progress="hidden",
        )
        save_case_btn.click(
            save_current_case,
            inputs=[current_user, current_result, case_note],
            outputs=[case_note, top_status_html, case_table, log_table],
            show_progress="hidden",
        )
        export_advice_btn.click(
            export_advice,
            inputs=[current_user, current_result],
            outputs=[advice_download, top_status_html, log_table],
            show_progress="hidden",
        )
        export_report_btn.click(
            export_report,
            inputs=[current_user, current_result],
            outputs=[report_download, top_status_html, log_table],
            show_progress="hidden",
        )

        refresh_inputs = [sample_keyword, sample_label, sample_source, case_keyword, case_label, case_rule_hit, case_rule_version, log_keyword, log_action, log_username]
        refresh_outputs = [sample_table, case_table, log_table, system_info_html]
        sample_search_btn.click(refresh_management_views, inputs=refresh_inputs, outputs=refresh_outputs, show_progress="hidden")
        case_search_btn.click(refresh_management_views, inputs=refresh_inputs, outputs=refresh_outputs, show_progress="hidden")
        log_search_btn.click(refresh_management_views, inputs=refresh_inputs, outputs=refresh_outputs, show_progress="hidden")
        refresh_btn.click(refresh_management_views, inputs=refresh_inputs, outputs=refresh_outputs, show_progress="hidden")

        sample_table.select(
            replay_sample_record,
            inputs=[sample_table],
            outputs=[input_mode, manual_text, current_json, summary_html, top_status_html, dataset_box, manual_box, upload_box],
        )
        case_table.select(
            replay_expert_case,
            inputs=[case_table],
            outputs=[input_mode, manual_text, current_json, summary_html, top_status_html, dataset_box, manual_box, upload_box],
        )

    return demo
