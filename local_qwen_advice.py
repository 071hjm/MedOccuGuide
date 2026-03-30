from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LocalQwenConfig:
    base_model_dir: Path
    adapter_dir: Path
    offload_dir: Path
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 1024
    max_input_tokens: int = 3072


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class LocalQwenAdviceGenerator:
    def __init__(self, config: LocalQwenConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._backend_error: str | None = None
        self._load_notice: str | None = None

    def _import_backend(self):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(f"本地大模型依赖未就绪：{exc}") from exc
        return torch, AutoTokenizer, AutoModelForCausalLM, PeftModel

    def _resolve_adapter_dir(self) -> Path:
        if (self.config.adapter_dir / "adapter_config.json").exists():
            return self.config.adapter_dir
        candidates = sorted(
            [
                path.parent
                for path in self.config.adapter_dir.rglob("adapter_config.json")
                if "checkpoint-" not in str(path.parent)
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"未找到 LoRA 适配器目录：{self.config.adapter_dir}")
        return candidates[0]

    def _resolve_base_model_dir(self, adapter_dir: Path) -> Path:
        if (self.config.base_model_dir / "config.json").exists():
            return self.config.base_model_dir
        adapter_config_path = adapter_dir / "adapter_config.json"
        if adapter_config_path.exists():
            payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            candidate = Path(str(payload.get("base_model_name_or_path", "")))
            if (candidate / "config.json").exists():
                return candidate
        raise RuntimeError(f"未找到基础模型目录：{self.config.base_model_dir}")

    def status(self) -> tuple[bool, str]:
        try:
            adapter_dir = self._resolve_adapter_dir()
            base_model_dir = self._resolve_base_model_dir(adapter_dir)
            self._import_backend()
            return True, f"本地模型可用：{base_model_dir} + {adapter_dir.name}"
        except Exception as exc:
            return False, str(exc)

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model, self._load_notice or "已复用本地模型"
        if self._backend_error:
            raise RuntimeError(self._backend_error)

        try:
            torch, AutoTokenizer, AutoModelForCausalLM, PeftModel = self._import_backend()
            adapter_dir = self._resolve_adapter_dir()
            base_model_dir = self._resolve_base_model_dir(adapter_dir)
            self.config.offload_dir.mkdir(parents=True, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_dir),
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    dtype=torch.bfloat16,
                )
                load_notice = "已加载本地 Qwen2.5-7B-Instruct（CPU bf16 模式）"
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_dir),
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    dtype="auto",
                    offload_folder=str(self.config.offload_dir),
                    offload_state_dict=True,
                )
                load_notice = "已加载本地 Qwen2.5-7B-Instruct，并启用自动设备映射/磁盘卸载"

            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                local_files_only=True,
                is_trainable=False,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._load_notice = load_notice
            return tokenizer, model, load_notice
        except Exception as exc:  # pragma: no cover - depends on local env
            self._backend_error = str(exc)
            raise

    @staticmethod
    def _render_record_block(record: dict[str, Any], max_items: int = 24) -> str:
        lines: list[str] = []
        for key, value in record.items():
            value_text = _clean_text(value)
            if value_text:
                lines.append(f"- {key}: {value_text}")
            if len(lines) >= max_items:
                break
        return "\n".join(lines) if lines else "- 无"

    @staticmethod
    def _render_evidence_block(final_result: dict[str, Any]) -> str:
        if final_result.get("source") == "expert_system":
            trace = final_result.get("trace") or []
            return "\n".join(f"- {item}" for item in trace[:8]) or "- 未提供规则路径"
        features = final_result.get("top_features") or []
        if not features:
            return "- 未提供 SHAP/贡献度证据"
        rows = []
        for item in features[:6]:
            rows.append(
                f"- {item.get('feature')}: value={item.get('value')} contribution={float(item.get('contribution', 0.0)):.4f}"
            )
        return "\n".join(rows)

    @staticmethod
    def _render_rag_block(rag_result: dict[str, Any]) -> str:
        docs = rag_result.get("documents") or []
        if not docs:
            return "- 未检索到片段"
        rows = []
        for idx, doc in enumerate(docs[:5], start=1):
            title = _clean_text(doc.get("title") or "未命名片段")
            source = _clean_text(doc.get("source") or doc.get("source_file") or "")
            text = _clean_text(doc.get("text") or "")[:220]
            rows.append(f"{idx}. {title} | {source}\n{text}")
        return "\n\n".join(rows)

    def build_prompt(
        self,
        record: dict[str, Any],
        final_result: dict[str, Any],
        rag_result: dict[str, Any],
        query: str,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "你是职业健康体检主检医生助理。"
            "请基于主检结论、证据依据和检索到的法规/知识片段，输出审慎、可执行的主检建议。"
            "不要编造未提供的检查结果，不要和主检结论冲突。"
            "如果证据不足，请明确说明“建议结合临床和职业史进一步确认”。"
        )
        user_prompt = f"""
请根据以下信息生成“主检建议”。

输出要求：
1. 用中文输出。
2. 只输出建议正文，不要解释你的推理过程。
3. 必须严格保留以下 4 个标题，并逐项填写具体内容：
   一、岗位处置建议：
   二、复查/转诊建议：
   三、职业防护与随访建议：
   四、提示说明：
4. 每部分 1-2 句，避免空话，不能 4 段都重复同一句话。
5. 若证据不足，可在“提示说明”中写“建议结合临床和职业史进一步确认”，但不要四段都只写这一句。

主检结论：{_clean_text(final_result.get("label"))}
诊断提示：{_clean_text(final_result.get("diagnosis")) or "无"}
判定来源：{_clean_text(final_result.get("source"))}

判定依据：
{self._render_evidence_block(final_result)}

样本有效信息：
{self._render_record_block(record)}

RAG 检索查询：
{_clean_text(query)}

RAG 召回片段：
{self._render_rag_block(rag_result)}
""".strip()
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def generate(
        self,
        record: dict[str, Any],
        final_result: dict[str, Any],
        rag_result: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        tokenizer, model, load_notice = self._load()
        torch, _, _, _ = self._import_backend()
        messages = self.build_prompt(record, final_result, rag_result, query)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        target_device = getattr(model, "device", None)
        if target_device is not None and str(target_device) != "meta":
            model_inputs = {key: value.to(target_device) for key, value in model_inputs.items()}

        do_sample = self.config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.05,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_ids = generated[0][prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("本地模型返回了空结果")
        bad_markers = [
            "无法作答",
            "信息不明确",
            "告诉我您想聊些什么",
            "和外星人交流",
            "请提供更多信息",
        ]
        if len(text) < 40 or any(marker in text for marker in bad_markers):
            raise RuntimeError(f"本地模型输出质量不足：{text[:120]}")

        lines = _split_lines(text)
        if len(set(lines)) <= 1:
            raise RuntimeError(f"本地模型输出重复度过高：{text[:120]}")
        structured_markers = ["一、岗位处置建议", "二、复查/转诊建议", "三、职业防护与随访建议", "四、提示说明"]
        if sum(1 for marker in structured_markers if marker in text) < 2:
            raise RuntimeError(f"本地模型未按要求输出结构化建议：{text[:120]}")
        if not lines:
            lines = [text]
        return {
            "mode": "local_qwen_lora",
            "notice": f"{load_notice}，并已使用 LoRA 适配器生成建议",
            "text": text,
            "lines": lines,
        }

    def safe_generate(
        self,
        record: dict[str, Any],
        final_result: dict[str, Any],
        rag_result: dict[str, Any],
        query: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return self.generate(record, final_result, rag_result, query), None
        except Exception as exc:  # pragma: no cover - depends on local env
            return None, str(exc)
