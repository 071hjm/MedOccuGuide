from __future__ import annotations

import json
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import jieba
import joblib
import numpy as np
import pandas as pd
import shap
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from data_process import clean_data, convert_duration_to_months, normalize_special_values, process_markers
from local_qwen_advice import LocalQwenAdviceGenerator, LocalQwenConfig


LABEL_MAPPING = {0: "其他疾病或异常", 1: "目前未见异常", 2: "复查", 3: "职业禁忌证", 4: "疑似职业病"}
THREE_CLASS_LABELS = {0: "其他疾病或异常", 1: "目前未见异常", 2: "复查"}

PRIVACY_AND_TARGET_COLUMNS = {
    "姓名0", "体检编号1", "主检结论1527", "主检建议1528", "职业禁忌证名称1529",
    "疑似职业病名称1530", "体检机构1531", "体检日期1532", "报告出具日期1533", "报告日期1534",
}

GENDER_COL = "性别2"
AGE_COL = "年龄3"
JOB_COL = "工种21"
TOTAL_WORK_YEARS_COL = "总工龄22"
EXPOSURE_YEARS_COL = "接害工龄23"
HAZARD_COL = "接触危害因素25"
PROTECTION_COL = "防护措施29"
TREATMENT_COL = "治疗经过33"
RADIATION_TYPE_COL = "放射线种类41"
FAMILY_HISTORY_COL = "家族史42"
PERSONAL_HISTORY_COL = "个人史43"
SYMPTOM_COL = "自觉症状45"
OTHER_SYMPTOM_COL = "其他症状46"

SBP_COL = "收缩压结果242"
SBP_FLAG_COL = "收缩压合格标记244"
DBP_COL = "舒张压结果245"
DBP_FLAG_COL = "舒张压合格标记247"
ECG_COL = "心电图结果572"
LUNG_COL = "肺功能结果结果590"
LUNG_FLAG_COL = "肺功能结果合格标记592"
CHEST_COL = "胸部正位片结果593"
CHEST_FLAG_COL = "胸部正位片合格标记595"
CHEST_JUDGMENT_COL = "胸部正位片结果判定596"
SPEECH_FREQ_COL = "双耳语频平均听阈结果732"
SPEECH_FREQ_FLAG_COL = "双耳语频平均听阈合格标记734"
HIGH_FREQ_COL = "双耳高频平均听阈结果735"
HIGH_FREQ_FLAG_COL = "双耳高频平均听阈合格标记737"
GLUCOSE_COL = "血葡萄糖结果1185"

SUMMARY_LAYOUT = [
    ("基本信息", [GENDER_COL, AGE_COL, JOB_COL]),
    ("职业暴露", [TOTAL_WORK_YEARS_COL, EXPOSURE_YEARS_COL, HAZARD_COL, PROTECTION_COL]),
    ("关键检查", [SBP_COL, DBP_COL, GLUCOSE_COL, CHEST_JUDGMENT_COL, LUNG_COL, HIGH_FREQ_COL, SPEECH_FREQ_COL, ECG_COL]),
]

SUMMARY_LABELS = {
    GENDER_COL: "性别", AGE_COL: "年龄", JOB_COL: "工种", TOTAL_WORK_YEARS_COL: "总工龄",
    EXPOSURE_YEARS_COL: "接害工龄", HAZARD_COL: "接触危害因素", PROTECTION_COL: "防护措施",
    SBP_COL: "收缩压", DBP_COL: "舒张压", GLUCOSE_COL: "血糖", CHEST_JUDGMENT_COL: "胸片判断",
    LUNG_COL: "肺功能", HIGH_FREQ_COL: "高频听阈", SPEECH_FREQ_COL: "语频听阈", ECG_COL: "心电图",
}

BINARY_FEATURES = [FAMILY_HISTORY_COL, PERSONAL_HISTORY_COL, "其他44", SYMPTOM_COL, OTHER_SYMPTOM_COL, "既往病史疾病名称30", PROTECTION_COL]
FILL_RULES = {FAMILY_HISTORY_COL: "无", OTHER_SYMPTOM_COL: "无", PERSONAL_HISTORY_COL: "无", RADIATION_TYPE_COL: "无", TREATMENT_COL: "已成"}
DURATION_FIELDS = [TOTAL_WORK_YEARS_COL, EXPOSURE_YEARS_COL]
ABNORMAL_KEYWORDS = ["异常", "增多", "增粗", "不齐", "阻塞", "拘束", "降低", "不合格", "结节", "斑片", "改变", "损害"]


@dataclass
class RuntimeConfig:
    base_dir: Path = Path(r"D:\codex")
    model_path: Path = Path(r"D:\codex\saved_models\LightGBM.pkl")
    feature_info_path: Path = Path(r"D:\codex\processed_data\feature_info.pkl")
    rules_path: Path = Path(r"D:\codex\expert_rules.json")
    data_path: Path = Path(r"D:\codex\data.pkl")
    rag_dir: Path = Path(r"D:\codex\rag")
    demo_samples_path: Path = Path(r"D:\codex\demo_test_samples.json")
    qwen_base_model_dir: Path = Path(r"D:\codex\models\Qwen2.5-7B-Instruct")
    qwen_lora_dir: Path = Path(r"D:\codex\LLaMA-Factory\saves\Qwen2.5-7B-Instruct\lora\train_2025-11-12-18-23-54")
    qwen_offload_dir: Path = Path(r"D:\codex\model_offload\qwen2.5_7b_lora")
    embedding_model_name: str = "bge-base-zh-v1.5"
    dense_top_k: int = 7
    sparse_top_k: int = 7
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 1024


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else default


def is_non_empty(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "None"}


def clean_filename(filename: str) -> str:
    cleaned = filename
    for char in '/\\:*?"<>| \t\n\r':
        cleaned = cleaned.replace(char, "_")
    return cleaned


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def as_jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif pd.isna(value):
            continue
        else:
            serializable[key] = value
    return serializable


def tokenize_zh(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()] if normalized else []


def looks_normal(text: str) -> bool:
    text = normalize_whitespace(text)
    return not text or any(token in text for token in ["正常", "未见异常", "合格", "阴性", "无明显异常", "未见明显异常"])


class ExpertRuleEngine:
    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = sorted(rules, key=lambda item: item.get("priority", 0), reverse=True)

    def _evaluate_leaf(self, node: dict[str, Any], record: dict[str, Any]) -> tuple[bool, list[str]]:
        field = node["field"]
        operator = node["operator"]
        target = node.get("value")
        raw_value = record.get(field, "")
        numeric_value = safe_float(raw_value, default=float("nan"))
        text_value = normalize_whitespace(raw_value)
        success = False
        if operator == ">=":
            success = not math.isnan(numeric_value) and numeric_value >= float(target)
        elif operator == ">":
            success = not math.isnan(numeric_value) and numeric_value > float(target)
        elif operator == "<=":
            success = not math.isnan(numeric_value) and numeric_value <= float(target)
        elif operator == "<":
            success = not math.isnan(numeric_value) and numeric_value < float(target)
        elif operator == "==":
            success = text_value == normalize_whitespace(target)
        elif operator == "!=":
            success = text_value != normalize_whitespace(target)
        elif operator == "contains_any":
            success = any(normalize_whitespace(item) in text_value for item in target)
        elif operator == "contains":
            success = normalize_whitespace(target) in text_value
        elif operator == "not_empty":
            success = is_non_empty(raw_value)
        return success, [f"{field} {operator} {target} | 当前值: {raw_value}"]

    def _evaluate_node(self, node: dict[str, Any], record: dict[str, Any]) -> tuple[bool, list[str]]:
        if "all" in node:
            lines: list[str] = []
            for child in node["all"]:
                child_success, child_lines = self._evaluate_node(child, record)
                lines.extend(child_lines)
                if not child_success:
                    return False, lines
            return True, lines
        if "any" in node:
            lines: list[str] = []
            for child in node["any"]:
                child_success, child_lines = self._evaluate_node(child, record)
                lines.extend(child_lines)
                if child_success:
                    return True, lines
            return False, lines
        return self._evaluate_leaf(node, record)

    def evaluate(self, record: dict[str, Any]) -> dict[str, Any] | None:
        for rule in self.rules:
            matched, trace = self._evaluate_node(rule["decision_tree"], record)
            if matched:
                return {"source": "expert_system", "label": rule["output_label"], "diagnosis": rule["diagnosis"], "rule_id": rule["id"], "trace": trace}
        return None


class HybridRetriever:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.documents = self._load_documents()
        self.embedding_backend = f"fallback_tfidf_for_{self.config.embedding_model_name}"
        if not self.documents:
            self.tokenized_docs = []
            self.bm25 = None
            self.vectorizer = None
            self.index = None
            return
        self.tokenized_docs = [tokenize_zh(doc["search_text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=12000)
        dense_matrix = self.vectorizer.fit_transform(doc["search_text"] for doc in self.documents).astype(np.float32)
        dense_embeddings = normalize(dense_matrix.toarray(), norm="l2").astype(np.float32)
        self.index = faiss.IndexFlatIP(dense_embeddings.shape[1])
        self.index.add(dense_embeddings)

    @staticmethod
    def _coerce_terms(value: Any) -> list[str]:
        if isinstance(value, list):
            return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
        text = normalize_whitespace(value)
        return [text] if text else []

    def _load_json_like_file(self, path: Path) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
            return docs
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    def _build_search_text(self, item: dict[str, Any], raw_text: str) -> str:
        pieces: list[str] = [
            item.get("title", ""), item.get("source", ""), item.get("hazard", ""), item.get("type", ""),
            item.get("source_type", ""), item.get("metric_name", ""), item.get("system_target", ""),
            item.get("retrieval_hint", ""), item.get("text", ""), item.get("dense_text", ""), raw_text,
        ]
        pieces.extend(self._coerce_terms(item.get("topic")))
        pieces.extend(self._coerce_terms(item.get("keywords")))
        pieces.extend(self._coerce_terms(item.get("sparse_terms")))
        return " ".join(piece for piece in (normalize_whitespace(v) for v in pieces) if piece)

    def _load_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(self.config.rag_dir.glob("*")):
            if path.suffix not in {".jsonl", ".json"}:
                continue
            for idx, item in enumerate(self._load_json_like_file(path)):
                raw_text = item.get("dense_text") or item.get("text") or json.dumps(item, ensure_ascii=False)
                docs.append(
                    {
                        "doc_id": item.get("chunk_id") or f"{path.stem}_{idx}",
                        "source_file": path.name,
                        "title": item.get("title") or item.get("metric_name") or path.stem,
                        "source": item.get("source") or path.name,
                        "text": item.get("text") or raw_text,
                        "dense_text": item.get("dense_text") or raw_text,
                        "search_text": self._build_search_text(item, raw_text),
                        "meta": item,
                    }
                )
        return docs

    @staticmethod
    def _normalize_score_map(pairs: list[tuple[int, float]]) -> dict[int, float]:
        if not pairs:
            return {}
        values = np.asarray([score for _, score in pairs], dtype=float)
        v_min = float(values.min())
        v_max = float(values.max())
        if abs(v_max - v_min) < 1e-9:
            return {idx: 1.0 for idx, _ in pairs}
        return {idx: float((score - v_min) / (v_max - v_min)) for idx, score in pairs}

    @staticmethod
    def _reference_matches_query(doc: dict[str, Any], query: str) -> bool:
        metric_name = normalize_whitespace(doc["meta"].get("metric_name") or doc["meta"].get("system_target") or "")
        return bool(metric_name) and (metric_name in query or any(token in query for token in ["参考范围", "阈值", "正常值", "异常值"]))

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        query = normalize_whitespace(query)
        top_k = top_k or self.config.dense_top_k
        if not query or not self.documents:
            return {"backend": self.embedding_backend, "dense_ranked": [], "sparse_ranked": [], "documents": [], "query_variants": []}

        query_variants = [query]
        query_variants.extend(normalize_whitespace(part) for part in re.split(r"[；;。.\n]", query) if normalize_whitespace(part))
        query_variants = list(dict.fromkeys(query_variants))[:6]
        query_tokens = set(tokenize_zh(query))
        search_k = min(max(top_k * 6, 20), len(self.documents))
        dense_best: dict[int, float] = {}
        sparse_best: dict[int, float] = {}

        for variant in query_variants:
            dense_query = self.vectorizer.transform([variant]).astype(np.float32).toarray()
            dense_query = normalize(dense_query, norm="l2").astype(np.float32)
            dense_scores, dense_indices = self.index.search(dense_query, search_k)
            for idx, score in zip(dense_indices[0], dense_scores[0]):
                if idx >= 0:
                    dense_best[int(idx)] = max(dense_best.get(int(idx), -1.0), float(score))
            sparse_tokens = tokenize_zh(variant)
            if sparse_tokens:
                bm25_scores = self.bm25.get_scores(sparse_tokens)
                sparse_order = np.argsort(bm25_scores)[::-1][:search_k]
                for idx in sparse_order:
                    sparse_best[int(idx)] = max(sparse_best.get(int(idx), -1e9), float(bm25_scores[idx]))

        dense_ranked = sorted(dense_best.items(), key=lambda item: item[1], reverse=True)
        sparse_ranked = sorted(sparse_best.items(), key=lambda item: item[1], reverse=True)
        dense_norm = self._normalize_score_map(dense_ranked[:search_k])
        sparse_norm = self._normalize_score_map(sparse_ranked[:search_k])

        reranked: list[tuple[int, float]] = []
        for idx in set(dense_norm) | set(sparse_norm):
            doc = self.documents[idx]
            meta = doc["meta"]
            score = 0.58 * dense_norm.get(idx, 0.0) + 0.42 * sparse_norm.get(idx, 0.0)
            score += min(len(set(tokenize_zh(doc["search_text"])) & query_tokens), 8) * 0.01
            score += 0.012 * len(set(tokenize_zh(doc["title"])) & query_tokens)
            hazard = normalize_whitespace(meta.get("hazard"))
            if hazard and hazard != "通用" and hazard in query:
                score += 0.16
            if meta.get("source_type") in {"national_standard_text", "national_standard_scan"}:
                score += 0.05
            if meta.get("type") == "reference_range_rule":
                score += 0.22 if self._reference_matches_query(doc, query) else -0.12
            metric_name = normalize_whitespace(meta.get("metric_name") or meta.get("system_target") or "")
            if metric_name and metric_name in query:
                score += 0.18
            joined = f"{doc['title']} {doc['text']}"
            if "职业禁忌证" in query and any(token in joined for token in ["职业禁忌证", "禁忌证"]):
                score += 0.14
            if "疑似职业病" in query and any(token in joined for token in ["疑似职业病", "界定", "诊断"]):
                score += 0.14
            if "复查" in query and any(token in joined for token in ["复查", "随访", "进一步检查"]):
                score += 0.08
            reranked.append((idx, score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        docs: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        reference_count = 0
        for idx, score in reranked:
            doc = self.documents[idx]
            meta = doc["meta"]
            dedupe_key = (doc["title"], doc["source"], str(meta.get("type") or ""))
            if dedupe_key in seen:
                continue
            if meta.get("type") == "reference_range_rule":
                if reference_count >= 2 or not self._reference_matches_query(doc, query):
                    continue
                reference_count += 1
            seen.add(dedupe_key)
            enriched = dict(doc)
            enriched["retrieval_score"] = float(score)
            docs.append(enriched)
            if len(docs) >= top_k:
                break

        return {"backend": self.embedding_backend, "dense_ranked": dense_ranked[: max(top_k * 3, 10)], "sparse_ranked": sparse_ranked[: max(top_k * 3, 10)], "documents": docs, "query_variants": query_variants}


class OccupationalHealthRuntime:
    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        self.feature_info = joblib.load(self.config.feature_info_path)
        self.model_bundle = joblib.load(self.config.model_path)
        self.pipeline = self.model_bundle["model"]
        self.scaler = self.pipeline.named_steps["scaler"]
        self.classifier = self.pipeline.named_steps["clf"]
        self.model_feature_names = self.model_bundle.get("feature_names") or self.feature_info["all_features"]
        self.rules = json.loads(self.config.rules_path.read_text(encoding="utf-8"))
        self.expert_engine = ExpertRuleEngine(self.rules)
        self.retriever = HybridRetriever(self.config)
        self._encoders: dict[str, Any | None] = {}
        self._raw_data: pd.DataFrame | None = None
        self._demo_samples_cache: list[dict[str, Any]] | None = None
        self._shap_explainer: Any | None = None
        self._local_advice_generator = LocalQwenAdviceGenerator(
            LocalQwenConfig(
                base_model_dir=self.config.qwen_base_model_dir,
                adapter_dir=self.config.qwen_lora_dir,
                offload_dir=self.config.qwen_offload_dir,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
            )
        )

    def _load_raw_data(self) -> pd.DataFrame:
        if self._raw_data is None:
            with self.config.data_path.open("rb") as handle:
                self._raw_data = pickle.load(handle)
        return self._raw_data

    def _load_encoder(self, column_name: str):
        if column_name in self._encoders:
            return self._encoders[column_name]
        path = self.config.base_dir / "saved_models" / f"label_encoder_{clean_filename(column_name)}.pkl"
        self._encoders[column_name] = joblib.load(path) if path.exists() else None
        return self._encoders[column_name]

    def _safe_encode_value(self, column_name: str, value: Any) -> int | float:
        encoder = self._load_encoder(column_name)
        if encoder is None:
            return safe_float(value, 0.0)
        text = normalize_whitespace(value)
        classes = [str(item) for item in encoder.classes_]
        if text in classes:
            return int(encoder.transform([text])[0])
        for fallback in ["无", "未见异常", "正常", "合格", "未检", "未检查", "其他", ""]:
            if fallback in classes:
                return int(encoder.transform([fallback])[0])
        return 0

    def _create_fixed_hazard_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if HAZARD_COL not in data.columns:
            data[HAZARD_COL] = ""
        hazards = data[HAZARD_COL].fillna("").astype(str)
        for hazard_col in [col for col in self.model_feature_names if col.startswith("危害因素_")]:
            hazard = hazard_col.replace("危害因素_", "", 1)
            data[hazard_col] = hazards.apply(lambda cell, name=hazard: 1 if name in [item.strip() for item in str(cell).replace("、", ",").split(",") if item.strip()] else 0)
        return data.drop(columns=[HAZARD_COL], errors="ignore")

    def build_model_input(self, record: dict[str, Any]) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = pd.DataFrame([record]).copy()
            row = normalize_special_values(row)
            row = row.fillna(FILL_RULES)
            for col in BINARY_FEATURES:
                if col in row.columns:
                    row[col] = (~row[col].astype(str).str.contains("无|未见|阴性", na=False)).astype(int)
            for col in DURATION_FIELDS:
                if col in row.columns:
                    row[col] = row[col].astype(str).apply(convert_duration_to_months)
            row = self._create_fixed_hazard_dummies(row)
            cleaned, _ = clean_data(row)
            cleaned = process_markers(cleaned)
            for col in self.model_feature_names:
                if col not in cleaned.columns:
                    cleaned[col] = 0 if col in self.feature_info["numeric_cols"] else ""
            cleaned = cleaned[self.model_feature_names].copy()
            for col in self.feature_info["numeric_cols"]:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)
            for col in self.feature_info["categorical_cols"]:
                cleaned[col] = cleaned[col].fillna("").astype(str).map(lambda value, name=col: self._safe_encode_value(name, value))
            return cleaned.astype(float)

    def build_demo_samples(self, per_label: int = 4) -> list[dict[str, Any]]:
        if self._demo_samples_cache is not None:
            return self._demo_samples_cache
        if self.config.demo_samples_path.exists():
            cached = json.loads(self.config.demo_samples_path.read_text(encoding="utf-8"))
            if cached:
                self._demo_samples_cache = cached
                return cached
        df = self._load_raw_data()
        collected: list[dict[str, Any]] = []
        for label in ["职业禁忌证", "疑似职业病", "复查", "其他疾病或异常", "目前未见异常"]:
            subset = df[df["主检结论1527"].astype(str) == label].copy()
            if subset.empty:
                continue
            subset["_field_score"] = subset.notna().sum(axis=1)
            candidates = []
            for idx, row in subset.sort_values("_field_score", ascending=False).head(200).iterrows():
                record = row.drop(labels=[col for col in PRIVACY_AND_TARGET_COLUMNS if col in row.index]).dropna().to_dict()
                expert_hit = self.expert_engine.evaluate(record)
                candidates.append({
                    "sample_id": f"{label}_{idx}",
                    "row_index": int(idx),
                    "expected_label": label,
                    "expected_diagnosis": str(row.get("职业禁忌证名称1529") or row.get("疑似职业病名称1530") or ""),
                    "display_name": f"{label} | {row.get(JOB_COL, '未知工种')} | 行 {idx}",
                    "record": as_jsonable_record(record),
                    "expert_hit": expert_hit["label"] if expert_hit else "",
                    "expert_diagnosis": expert_hit["diagnosis"] if expert_hit else "",
                    "field_score": int(row["_field_score"]),
                })
            candidates.sort(key=lambda item: (0 if item["expert_hit"] == label else 1, -item["field_score"]) if label in {"职业禁忌证", "疑似职业病"} else (-item["field_score"]))
            collected.extend(candidates[:per_label])
        self.config.demo_samples_path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8")
        self._demo_samples_cache = collected
        return collected

    def get_sample_by_id(self, sample_id: str) -> dict[str, Any] | None:
        return next((sample for sample in self.build_demo_samples() if sample["sample_id"] == sample_id), None)

    @staticmethod
    def _collect_summary_highlights(record: dict[str, Any]) -> list[str]:
        items: list[str] = []
        if safe_float(record.get(SBP_COL), 0) >= 140 or safe_float(record.get(DBP_COL), 0) >= 90:
            items.append(f"血压偏高：{record.get(SBP_COL, '')}/{record.get(DBP_COL, '')}")
        if safe_float(record.get(GLUCOSE_COL), 0) >= 6.1:
            items.append(f"血糖升高：{record.get(GLUCOSE_COL, '')}")
        if is_non_empty(record.get(CHEST_JUDGMENT_COL)) and not looks_normal(str(record.get(CHEST_JUDGMENT_COL))):
            items.append(f"胸片提示：{record.get(CHEST_JUDGMENT_COL)}")
        if is_non_empty(record.get(LUNG_COL)) and any(token in str(record.get(LUNG_COL)) for token in ABNORMAL_KEYWORDS):
            items.append(f"肺功能异常提示：{record.get(LUNG_COL)}")
        if safe_float(record.get(HIGH_FREQ_COL), 0) >= 40:
            items.append(f"高频听阈异常：{record.get(HIGH_FREQ_COL)}")
        if safe_float(record.get(SPEECH_FREQ_COL), 0) >= 25:
            items.append(f"语频听阈异常：{record.get(SPEECH_FREQ_COL)}")
        if is_non_empty(record.get(ECG_COL)) and not looks_normal(str(record.get(ECG_COL))):
            items.append(f"心电图提示：{record.get(ECG_COL)}")
        if is_non_empty(record.get(SYMPTOM_COL)) and not looks_normal(str(record.get(SYMPTOM_COL))):
            items.append(f"自觉症状：{record.get(SYMPTOM_COL)}")
        return items[:6]

    @staticmethod
    def summarize_record(record: dict[str, Any], source_note: str = "") -> str:
        sections: list[str] = []
        for title, fields in SUMMARY_LAYOUT:
            cells = [f"<div><span>{SUMMARY_LABELS.get(field, field)}：</span>{record.get(field)}</div>" for field in fields if is_non_empty(record.get(field))]
            if cells:
                sections.append(
                    f"<div class='summary-section'><div class='summary-section-title'>{title}</div>"
                    f"<div class='summary-grid'>{''.join(cells)}</div></div>"
                )
        highlights = OccupationalHealthRuntime._collect_summary_highlights(record)
        if highlights:
            sections.append(
                f"<div class='summary-section summary-section-wide'><div class='summary-section-title'>重点异常</div>"
                f"<ul class='summary-warning-list'>{''.join(f'<li>{item}</li>' for item in highlights)}</ul></div>"
            )
        hazard_text = normalize_whitespace(record.get(HAZARD_COL))
        if hazard_text:
            tags = [item.strip() for item in re.split(r'[,、]', hazard_text) if item.strip()]
            sections.append(
                f"<div class='summary-section summary-section-wide'><div class='summary-section-title'>暴露标签</div>"
                f"<div class='summary-tags'>{''.join(f'<span>{tag}</span>' for tag in tags[:8])}</div></div>"
            )
        if not sections:
            sections.append("<div class='summary-empty'>当前记录字段较少，暂无可展示摘要。</div>")
        note_html = f"<div class='summary-note'>{source_note}</div>" if source_note else ""
        return (
            "<div class='summary-shell'>"
            "<div class='field-label'>样本摘要</div>"
            f"{note_html}"
            f"<div class='summary-sections-grid'>{''.join(sections)}</div>"
            "</div>"
        )

    def _heuristic_three_class(self, record: dict[str, Any]) -> dict[str, Any]:
        abnormal_score = len(self._collect_summary_highlights(record))
        hazard_text = normalize_whitespace(record.get(HAZARD_COL))
        occupational_signal = any(token in hazard_text for token in ["噪声", "粉尘", "苯", "高温", "射线"])
        label = "目前未见异常" if abnormal_score == 0 else ("复查" if occupational_signal else "其他疾病或异常")
        return {"source": "heuristic_fallback", "label": label, "probabilities": {}, "top_features": [], "trace": ["LightGBM 执行异常时启用保底启发式规则。"]}

    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.classifier)
        return self._shap_explainer

    def _extract_shap_evidence(self, model_input: pd.DataFrame, predicted_index: int) -> tuple[list[dict[str, Any]], str]:
        scaled = self.scaler.transform(model_input.values.astype(float))
        try:
            shap_values = self._get_shap_explainer().shap_values(scaled)
            shap_array = np.asarray(shap_values)
            contributions = shap_array[0, :, predicted_index] if shap_array.ndim == 3 else shap_array[0]
            mode = "shap"
        except Exception:
            contrib = np.asarray(self.classifier.booster_.predict(scaled, pred_contrib=True))
            n_features = model_input.shape[1]
            contributions = contrib[0, predicted_index * (n_features + 1): predicted_index * (n_features + 1) + n_features]
            mode = "lightgbm_pred_contrib"
        items = [{"feature": feature, "value": model_input.iloc[0].to_dict().get(feature), "contribution": float(value)} for feature, value in zip(self.model_feature_names, contributions)]
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return items[:8], mode

    def ml_predict(self, record: dict[str, Any]) -> dict[str, Any]:
        try:
            model_input = self.build_model_input(record)
            probabilities = self.pipeline.predict_proba(model_input.values.astype(float))[0]
            pred_idx = int(np.argmax(probabilities))
            features, explain_mode = self._extract_shap_evidence(model_input, pred_idx)
            return {"source": "lightgbm", "label": THREE_CLASS_LABELS[pred_idx], "probabilities": {THREE_CLASS_LABELS[i]: float(p) for i, p in enumerate(probabilities)}, "top_features": features, "explain_mode": explain_mode}
        except Exception as exc:
            fallback = self._heuristic_three_class(record)
            fallback["error"] = str(exc)
            return fallback

    def build_query_text(self, record: dict[str, Any], final_result: dict[str, Any]) -> str:
        parts = [f"主检结论 {final_result['label']}"]
        if final_result.get("diagnosis"):
            parts.append(f"诊断提示 {final_result['diagnosis']}")
        for field in [JOB_COL, HAZARD_COL, CHEST_JUDGMENT_COL, LUNG_COL, ECG_COL]:
            value = normalize_whitespace(record.get(field))
            if value:
                parts.append(f"{SUMMARY_LABELS.get(field, field)} {value}")
        parts.extend(self._collect_summary_highlights(record))
        return "；".join(parts[:12])

    def _detect_local_llm_notice(self) -> str:
        adapter_path = self.config.base_dir / "LLaMA-Factory" / "saves" / "Qwen2.5-7B-Instruct" / "lora" / "train_2025-11-12-18-23-54" / "adapter_config.json"
        if self.config.qwen_base_model_dir.exists() and (self.config.qwen_base_model_dir / "config.json").exists():
            return f"已检测到本地基础模型：{self.config.qwen_base_model_dir}"
        if adapter_path.exists():
            adapter_config = json.loads(adapter_path.read_text(encoding="utf-8"))
            base_model_path = Path(str(adapter_config.get("base_model_name_or_path", "")))
            if base_model_path.exists():
                return f"已检测到 LoRA 引用的基础模型：{base_model_path}"
        return "尚未检测到可直接推理的本地基础模型，当前使用模板保底生成建议。"

    def generate_advice(self, record: dict[str, Any], final_result: dict[str, Any], rag_result: dict[str, Any]) -> dict[str, Any]:
        if final_result["label"] == "职业禁忌证":
            employment = "建议暂缓继续从事相关职业病危害因素暴露岗位，并由主检医师结合专科检查结果进一步复核。"
        elif final_result["label"] == "疑似职业病":
            employment = "建议尽快进入职业病进一步诊断或复核流程，必要时临时调整离开相关危害暴露岗位。"
        elif final_result["label"] == "复查":
            employment = "建议在完成复查项目之前维持观察管理，复核后再确认主检结论。"
        else:
            employment = "当前未见必须立即调岗的直接证据，可结合岗位危害因素继续随访管理。"
        follow_up = self._collect_summary_highlights(record) or ["结合本次异常项安排常规复查与动态随访。"]
        prevention = ["持续落实岗位防尘、防噪声及个体防护用品佩戴。", "结合接触危害因素记录完善职业健康监护档案。", "对异常指标开展动态随访，避免仅凭单次结果做最终判断。"]
        references = [f"{doc['title']} | {doc['source']} | {normalize_whitespace(doc['text'])[:120]}" for doc in rag_result["documents"][:3]]
        notice = self._detect_local_llm_notice()
        lines = ["1. 岗位处置建议：", employment, "2. 推荐复查与随访：", "；".join(follow_up), "3. 职业防护建议：", "；".join(prevention), "4. RAG 依据片段：", "；".join(references) if references else "未检索到足够相关片段，已回退到通用建议模板。", "5. 模型状态：", notice]
        return {"mode": "template_fallback", "notice": notice, "text": "\n".join(lines), "lines": lines}

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        expert_result = self.expert_engine.evaluate(record)
        if expert_result:
            final_result, ml_result = expert_result, None
        else:
            ml_result = self.ml_predict(record)
            final_result = {"source": ml_result["source"], "label": ml_result["label"], "diagnosis": "", "trace": ml_result.get("trace", []), "probabilities": ml_result.get("probabilities", {}), "top_features": ml_result.get("top_features", []), "explain_mode": ml_result.get("explain_mode", "")}
        query = self.build_query_text(record, final_result)
        rag_result = self.retriever.search(query)
        advice = self.generate_advice(record, final_result, rag_result)
        return {"record": record, "final_result": final_result, "expert_result": expert_result, "ml_result": ml_result, "rag_result": rag_result, "advice": advice, "query": query}


def render_conclusion_html(result: dict[str, Any]) -> str:
    final_result = result["final_result"]
    diagnosis = final_result.get("diagnosis") or "未命名"
    source_label = "专家系统命中" if final_result["source"] == "expert_system" else "LightGBM/保底判定"
    return "<div class='alert-strip'>" f"<span>检测结论：{final_result['label']}</span>" f"<span>判定来源：{source_label} | 诊断提示：{diagnosis}</span>" "</div>"


def render_logic_box_html(result: dict[str, Any]) -> str:
    final_result = result["final_result"]
    items = final_result.get("trace", []) if final_result["source"] == "expert_system" else [*(f"{label}: {prob:.2%}" for label, prob in final_result.get("probabilities", {}).items()), f"解释方式：{final_result.get('explain_mode', 'heuristic_fallback')}"]
    return f"<div class='mini-box'><h3>规则路径 / 模型路径</h3><ul>{''.join(f'<li>{item}</li>' for item in items)}</ul></div>"


def render_evidence_box_html(result: dict[str, Any]) -> str:
    final_result = result["final_result"]
    rows = [f"命中专家规则：{final_result.get('rule_id', '')}", f"规则判定疾病：{final_result.get('diagnosis', '')}"] if final_result["source"] == "expert_system" else [f"{item['feature']} = {item['value']} | 贡献值 = {item['contribution']:.4f}" for item in final_result.get("top_features", [])[:6]]
    if final_result["source"] == "heuristic_fallback":
        rows.append("LightGBM 或 SHAP 执行异常，当前显示保底启发式依据。")
    return f"<div class='mini-box'><h3>异常项与判定依据</h3><ul>{''.join(f'<li>{row}</li>' for row in rows if row)}</ul></div>"


def render_suggestion_html(result: dict[str, Any]) -> str:
    advice = result["advice"]
    return f"<div class='suggestions-body'>{''.join(f'<p>{line}</p>' for line in advice['lines'])}</div>"


def render_process_html(result: dict[str, Any]) -> str:
    steps = [f"样本输入：已接收 {len(result['record'])} 个有效字段。", f"专家系统：{'命中' if result['expert_result'] else '未命中'}。", f"ML 兜底：{'已执行' if result['ml_result'] else '未执行'}。", f"RAG 检索：向量召回 + BM25，最终返回 {len(result['rag_result']['documents'])} 条片段。", f"查询改写：{' | '.join(result['rag_result'].get('query_variants', [])[:4])}", f"建议生成：{result['advice']['notice']}"]
    docs = [f"{doc['title']} | {doc['source']} | {normalize_whitespace(doc['text'])[:120]}" for doc in result["rag_result"]["documents"][:5]]
    return "<div class='stage-pane-content'>" f"<ol class='process-list'>{''.join(f'<li>{line}</li>' for line in steps)}</ol>" "<div class='process-divider'></div>" "<div class='process-source-title'>RAG 命中片段</div>" f"<ul class='process-source-list'>{''.join(f'<li>{line}</li>' for line in docs)}</ul>" "</div>"


def build_report_text(result: dict[str, Any]) -> str:
    final_result = result["final_result"]
    record = result["record"]
    summary_lines = [f"{key}: {value}" for key, value in record.items() if is_non_empty(value)]
    logic_lines = final_result.get("trace", []) if final_result["source"] == "expert_system" else [f"{item['feature']}={item['value']} | contribution={item['contribution']:.4f}" for item in final_result.get("top_features", [])]
    rag_lines = [f"{doc['title']} | {doc['source']} | score={doc.get('retrieval_score', 0):.4f}" for doc in result["rag_result"]["documents"]]
    return "\n".join(["职业健康主检智能推理报告", "", "一、样本有效信息", *summary_lines, "", "二、主检结论", f"结论: {final_result['label']}", f"来源: {final_result['source']}", f"诊断提示: {final_result.get('diagnosis', '')}", "", "三、判定依据", *logic_lines, "", "四、RAG 片段", *rag_lines, "", "五、主检建议", result["advice"]["text"]])


@lru_cache(maxsize=1)
def get_runtime() -> OccupationalHealthRuntime:
    return OccupationalHealthRuntime(RuntimeConfig())
