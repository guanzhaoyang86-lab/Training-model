from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import GroundingResult
from .taxonomy import TYPE_NAMES, canonicalize_type_name
from .utils import dump_json


METRIC_KEYS = (
    "parse_success_rate",
    "exact_match",
    "qa_exact_match",
    "grounding_exact_match",
    "summary_exact_match",
    "type_accuracy",
    "strength_accuracy",
    "direction_accuracy",
    "start_mae",
    "end_mae",
    "point_precision",
    "point_recall",
    "point_f1",
    "mean_iou",
)


def parse_generated_result(text: str) -> GroundingResult:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("未在模型输出中找到合法 JSON 对象。")
    payload = json.loads(text[start : end + 1])
    return GroundingResult.from_dict(payload)


def _interval_to_mask(items: list[dict[str, Any]], series_length: int) -> list[int]:
    mask = [0 for _ in range(series_length)]
    for item in items:
        start = int(item["start"])
        end = int(item["end"])
        start = max(0, min(series_length - 1, start))
        end = max(start, min(series_length - 1, end))
        for idx in range(start, end + 1):
            mask[idx] = 1
    return mask


def _point_prf(gt_mask: list[int], pred_mask: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for gt, pred in zip(gt_mask, pred_mask) if gt == 1 and pred == 1)
    fp = sum(1 for gt, pred in zip(gt_mask, pred_mask) if gt == 0 and pred == 1)
    fn = sum(1 for gt, pred in zip(gt_mask, pred_mask) if gt == 1 and pred == 0)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _interval_iou(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]) -> float:
    if not gt_items and not pred_items:
        return 1.0
    if not gt_items or not pred_items:
        return 0.0

    def _union_length(items: list[dict[str, Any]]) -> int:
        merged: list[list[int]] = []
        for item in sorted(items, key=lambda value: (int(value["start"]), int(value["end"]))):
            start = int(item["start"])
            end = int(item["end"])
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return sum(end - start + 1 for start, end in merged)

    intersections: list[dict[str, int]] = []
    for gt in gt_items:
        gt_start = int(gt["start"])
        gt_end = int(gt["end"])
        for pred in pred_items:
            pred_start = int(pred["start"])
            pred_end = int(pred["end"])
            left = max(gt_start, pred_start)
            right = min(gt_end, pred_end)
            if left <= right:
                intersections.append({"start": left, "end": right})

    inter = _union_length(intersections)
    union = _union_length(gt_items) + _union_length(pred_items) - inter
    return 0.0 if union <= 0 else inter / union


def _average_item_accuracy(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]], key: str) -> float:
    if not gt_items and not pred_items:
        return 1.0
    if not gt_items or not pred_items:
        return 0.0

    total = max(len(gt_items), len(pred_items))
    correct = 0
    for index in range(total):
        if index >= len(gt_items) or index >= len(pred_items):
            continue
        correct += float(gt_items[index][key] == pred_items[index][key])
    return float(correct / total)


def _average_boundary_mae(
    gt_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    *,
    key: str,
    series_length: int,
) -> float:
    if not gt_items and not pred_items:
        return 0.0
    if not gt_items or not pred_items:
        return float(series_length)

    total = max(len(gt_items), len(pred_items))
    error = 0.0
    for index in range(total):
        if index >= len(gt_items) or index >= len(pred_items):
            error += float(series_length)
            continue
        error += abs(int(gt_items[index][key]) - int(pred_items[index][key]))
    return float(error / total)


def score_prediction(
    *,
    target: GroundingResult,
    prediction: GroundingResult,
    series_length: int,
) -> dict[str, float]:
    target_dict = target.to_dict()
    prediction_dict = prediction.to_dict()
    gt_items = target_dict["evidence"]
    pred_items = prediction_dict["evidence"]

    type_ok = 0.0
    strength_ok = 0.0
    direction_ok = 0.0
    start_mae = float(series_length)
    end_mae = float(series_length)

    if gt_items and pred_items:
        type_ok = _average_item_accuracy(gt_items, pred_items, "type")
        strength_ok = _average_item_accuracy(gt_items, pred_items, "strength")
        direction_ok = _average_item_accuracy(gt_items, pred_items, "direction")
        start_mae = _average_boundary_mae(gt_items, pred_items, key="start", series_length=series_length)
        end_mae = _average_boundary_mae(gt_items, pred_items, key="end", series_length=series_length)
    elif not gt_items and not pred_items:
        type_ok = 1.0
        strength_ok = 1.0
        direction_ok = 1.0
        start_mae = 0.0
        end_mae = 0.0

    gt_mask = _interval_to_mask(gt_items, series_length=series_length)
    pred_mask = _interval_to_mask(pred_items, series_length=series_length)
    precision, recall, f1 = _point_prf(gt_mask, pred_mask)

    return {
        "exact_match": float(target_dict == prediction_dict),
        "summary_exact_match": float(target.summary == prediction.summary),
        "type_accuracy": type_ok,
        "strength_accuracy": strength_ok,
        "direction_accuracy": direction_ok,
        "start_mae": start_mae,
        "end_mae": end_mae,
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": f1,
        "mean_iou": _interval_iou(gt_items, pred_items),
    }


def _normalize_exact_text(text: str) -> str:
    return " ".join(text.strip().split())


def _is_qa_record(record: dict[str, Any]) -> bool:
    return record.get("metadata", {}).get("task_type") == "qa" or "answer" in record.get("target", {})


def _new_metric_totals() -> dict[str, list[float]]:
    return {key: [] for key in METRIC_KEYS}


def _mean_metrics(totals: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(sum(values) / len(values)) if values else 0.0
        for key, values in totals.items()
    }


def _record_anomaly_types(record: dict[str, Any]) -> list[str]:
    raw_type = record.get("metadata", {}).get("anomaly_type")
    if isinstance(raw_type, str):
        canonical = canonicalize_type_name(raw_type)
        return [canonical] if canonical in TYPE_NAMES else []
    if isinstance(raw_type, list):
        names = [canonicalize_type_name(str(item)) for item in raw_type]
        return sorted({name for name in names if name in TYPE_NAMES})

    target = record.get("target", {})
    if not isinstance(target, dict):
        return []
    evidence = target.get("evidence", [])
    if not isinstance(evidence, list):
        return []
    names = []
    for item in evidence:
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            names.append(canonicalize_type_name(str(item["type"])))
    return sorted({name for name in names if name in TYPE_NAMES})


def _append_metric(
    *,
    totals: dict[str, list[float]],
    type_totals: dict[str, dict[str, list[float]]],
    type_names: list[str],
    key: str,
    value: float,
) -> None:
    totals[key].append(value)
    for type_name in type_names:
        type_totals[type_name][key].append(value)


def evaluate_predictions(dataset_records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pred_by_id = {item["id"]: item for item in predictions}
    totals = _new_metric_totals()
    type_totals = {type_name: _new_metric_totals() for type_name in TYPE_NAMES}
    type_counts = {
        type_name: {
            "num_examples": 0,
            "num_grounding_examples": 0,
            "num_qa_examples": 0,
        }
        for type_name in TYPE_NAMES
    }
    num_qa_examples = 0
    num_grounding_examples = 0

    for record in dataset_records:
        item_id = record["id"]
        type_names = _record_anomaly_types(record)
        for type_name in type_names:
            type_counts[type_name]["num_examples"] += 1
        prediction_info = pred_by_id.get(item_id)
        if prediction_info is None:
            if _is_qa_record(record):
                num_qa_examples += 1
                for type_name in type_names:
                    type_counts[type_name]["num_qa_examples"] += 1
                _append_metric(
                    totals=totals,
                    type_totals=type_totals,
                    type_names=type_names,
                    key="exact_match",
                    value=0.0,
                )
                _append_metric(
                    totals=totals,
                    type_totals=type_totals,
                    type_names=type_names,
                    key="qa_exact_match",
                    value=0.0,
                )
            else:
                num_grounding_examples += 1
                for type_name in type_names:
                    type_counts[type_name]["num_grounding_examples"] += 1
                for key in ("exact_match", "parse_success_rate", "grounding_exact_match"):
                    _append_metric(
                        totals=totals,
                        type_totals=type_totals,
                        type_names=type_names,
                        key=key,
                        value=0.0,
                    )
            continue
        if _is_qa_record(record):
            num_qa_examples += 1
            for type_name in type_names:
                type_counts[type_name]["num_qa_examples"] += 1
            target_answer = str(record.get("target", {}).get("answer", record.get("assistant_text", "")))
            pred_answer = str(prediction_info.get("generated_text", ""))
            exact = float(_normalize_exact_text(target_answer) == _normalize_exact_text(pred_answer))
            for key in ("exact_match", "qa_exact_match"):
                _append_metric(
                    totals=totals,
                    type_totals=type_totals,
                    type_names=type_names,
                    key=key,
                    value=exact,
                )
            continue

        num_grounding_examples += 1
        for type_name in type_names:
            type_counts[type_name]["num_grounding_examples"] += 1
        target = GroundingResult.from_dict(record["target"])
        series_length = int(record["metadata"]["series_length"])
        try:
            prediction = parse_generated_result(prediction_info["generated_text"])
        except Exception:
            for key in ("parse_success_rate", "exact_match", "grounding_exact_match"):
                _append_metric(
                    totals=totals,
                    type_totals=type_totals,
                    type_names=type_names,
                    key=key,
                    value=0.0,
                )
            continue
        _append_metric(
            totals=totals,
            type_totals=type_totals,
            type_names=type_names,
            key="parse_success_rate",
            value=1.0,
        )
        scored = score_prediction(target=target, prediction=prediction, series_length=series_length)
        _append_metric(
            totals=totals,
            type_totals=type_totals,
            type_names=type_names,
            key="grounding_exact_match",
            value=scored["exact_match"],
        )
        for key, value in scored.items():
            _append_metric(
                totals=totals,
                type_totals=type_totals,
                type_names=type_names,
                key=key,
                value=value,
            )

    return {
        "num_examples": len(dataset_records),
        "num_grounding_examples": num_grounding_examples,
        "num_qa_examples": num_qa_examples,
        "metrics": _mean_metrics(totals),
        "metrics_by_type": {
            type_name: {
                **type_counts[type_name],
                "metrics": _mean_metrics(type_totals[type_name]),
            }
            for type_name in TYPE_NAMES
        },
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def evaluate_prediction_files(
    *,
    dataset_jsonl: str | Path,
    predictions_jsonl: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_records = load_jsonl(dataset_jsonl)
    predictions = load_jsonl(predictions_jsonl)
    result = evaluate_predictions(dataset_records, predictions)
    if output_path is not None:
        dump_json(result, output_path)
    return result
