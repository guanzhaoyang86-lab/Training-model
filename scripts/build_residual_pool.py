from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.event_metrics import (  # noqa: E402
    DEFAULT_MAX_INDEX,
    DEFAULT_PREDICTION_SCHEMA,
    DEFAULT_SERIES_LENGTH,
    DEFAULT_TAU_GOOD,
    DEFAULT_TAU_MATCH,
    events_from_ground_truth,
    residual_pool_summary,
    score_prediction_output,
)
from ts_grounder.utils import load_yaml  # noqa: E402
from ts_grounder.vlm_data import _record_from_sample  # noqa: E402
from ts_grounder.vlm_eval import load_jsonl  # noqa: E402
from ts_grounder.vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_chat_messages  # noqa: E402


def _load_split_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path), "jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array or JSONL records")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} must contain JSON objects")
    return payload, "raw_json"


def _resolve_path(path: str | Path | None, *, base: Path) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _load_config(path: str | None, *, repo_root: Path) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    return load_yaml(config_path)


def _records_from_raw_samples(
    samples: list[dict[str, Any]],
    *,
    raw_split_path: Path,
    cfg: dict[str, Any],
    output_root: Path,
    source_root_arg: str | None,
    split: str | None,
) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt", {}), dict) else {}
    source_root = _resolve_path(source_root_arg, base=repo_root)
    if source_root is None:
        source_root = raw_split_path.parent.resolve()
    split_name = split or raw_split_path.stem

    records: list[dict[str, Any]] = []
    for sample in samples:
        record = _record_from_sample(
            sample,
            source_root=source_root,
            output_root=output_root,
            split=split_name,
            system_prompt=str(prompt_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            user_prompt=str(prompt_cfg.get("user_prompt", DEFAULT_USER_PROMPT)),
            image_subdir=data_cfg.get("image_subdir"),
            include_indexed_series_text=bool(data_cfg.get("include_indexed_series_text", False)),
            indexed_series_precision=int(data_cfg.get("indexed_series_precision", 4)),
            indexed_series_compact=bool(data_cfg.get("indexed_series_compact", False)),
            series_normalization=data_cfg.get("series_normalization"),
        )
        records.append(record)
    return records


def _record_ground_truth(record: dict[str, Any]) -> Any:
    if "ground_truth" in record:
        return record["ground_truth"]
    if "target" in record:
        return record["target"]
    raw_sample = record.get("raw_sample")
    if isinstance(raw_sample, dict):
        if "events" in raw_sample:
            return {"evidence": raw_sample.get("events", [])}
        if "point_labels" in raw_sample:
            return raw_sample.get("point_labels", [])
    for key in ("evidence", "events", "point_labels", "labels"):
        if key in record:
            return record[key]
    raise KeyError(f"record {record.get('id') or record.get('sample_id') or '<unknown>'} is missing ground truth")


def _ground_truth_summary(record: dict[str, Any]) -> str:
    for container_key in ("ground_truth", "target"):
        container = record.get(container_key)
        if isinstance(container, dict) and isinstance(container.get("summary"), str):
            return str(container["summary"])
    if not events_from_ground_truth(record):
        return "No anomaly detected."
    return "Ground-truth anomaly evidence is provided."


def _dataset_for_record(record: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source_dataset") not in (None, ""):
        return str(metadata["source_dataset"])
    raw_sample = record.get("raw_sample")
    if isinstance(raw_sample, dict):
        context = raw_sample.get("context")
        if isinstance(context, dict) and context.get("source_dataset") not in (None, ""):
            return str(context["source_dataset"])
    if record.get("dataset") not in (None, ""):
        return str(record["dataset"])
    return "unknown"


def _sample_id_for_record(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id") or record.get("metadata", {}).get("sample_id") or "")


def _resolve_image_path(record: dict[str, Any], image_root: str | None) -> Path:
    raw_path = record.get("image_path") or record.get("image_relpath")
    if raw_path is None:
        raise KeyError(f"record {_sample_id_for_record(record) or '<unknown>'} is missing image_path")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if image_root:
        return Path(image_root).expanduser() / path
    return path


def _resolve_dtype(value: str | None):
    if value in (None, "", "none"):
        return None
    import torch

    normalized = value.strip().lower()
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype_map[normalized]


def _move_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    assistant_markers = (
        "ASSISTANT:",
        "Assistant:",
        "<|assistant|>",
        "<|im_start|>assistant",
    )
    for marker in assistant_markers:
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[-1].strip()
    return cleaned


def _decode_new_tokens(processor, model_inputs: dict[str, Any], generated_ids) -> list[str]:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        return [
            _clean_generated_text(text)
            for text in processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]
    prompt_length = int(input_ids.shape[1])
    new_token_ids = generated_ids[:, prompt_length:]
    if new_token_ids.shape[1] == 0:
        return ["" for _ in range(int(generated_ids.shape[0]))]
    return [
        _clean_generated_text(text)
        for text in processor.batch_decode(new_token_ids, skip_special_tokens=True)
    ]


def _generate_outputs(
    *,
    model_path: str,
    records: list[dict[str, Any]],
    image_root: str | None,
    batch_size: int,
    max_new_tokens: int,
    torch_dtype: str | None,
    device_map: str | None,
) -> list[str]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {}
    resolved_dtype = _resolve_dtype(torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    if device_map:
        device = next(
            (param.device for param in model.parameters() if param.device.type != "meta"),
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    outputs: list[str] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = [Image.open(_resolve_image_path(record, image_root)).convert("RGB") for record in batch_records]
        prompt_texts = []
        for record in batch_records:
            messages = build_chat_messages(
                system_prompt=str(record["system_prompt"]),
                user_prompt=str(record["user_prompt"]),
            )
            prompt_texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        model_inputs = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        model_inputs = _move_to_device(model_inputs, device)
        generation_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)
        outputs.extend(_decode_new_tokens(processor, model_inputs, generated_ids))
        print(f"inference {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return outputs


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_residual_pool(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.real_train_data).expanduser()
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    residual_record_root = Path(args.record_output_root).expanduser() if args.record_output_root else output_path.parent / "residual_record_cache"
    if not residual_record_root.is_absolute():
        residual_record_root = repo_root / residual_record_root

    cfg = _load_config(args.config, repo_root=repo_root)
    loaded_records, input_kind = _load_split_records(input_path)
    if input_kind == "raw_json":
        records = _records_from_raw_samples(
            loaded_records,
            raw_split_path=input_path,
            cfg=cfg,
            output_root=residual_record_root,
            source_root_arg=args.source_root,
            split=args.split,
        )
    else:
        records = loaded_records
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("No records selected for residual mining")

    generated_outputs = _generate_outputs(
        model_path=args.model_path,
        records=records,
        image_root=args.image_root,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    residual_records: list[dict[str, Any]] = []
    for record, generated_text in zip(records, generated_outputs):
        ground_truth = _record_ground_truth(record)
        gt_evidence = events_from_ground_truth(ground_truth)
        scored = score_prediction_output(
            pred_output=generated_text,
            gt={"evidence": gt_evidence},
            tau_match=float(args.tau_match),
            tau_good=float(args.tau_good),
            series_length=int(args.series_length),
            max_index=int(args.max_index),
            prediction_schema=str(args.prediction_schema),
        )
        sample_id = _sample_id_for_record(record)
        dataset = _dataset_for_record(record, args.dataset_name)
        residual_record = dict(record)
        residual_record["id"] = residual_record.get("id") or sample_id
        residual_record["sample_id"] = sample_id
        residual_record["dataset"] = dataset
        residual_record["ground_truth"] = {
            "evidence": gt_evidence,
            "summary": _ground_truth_summary(record),
        }
        residual_record["gt_evidence"] = gt_evidence
        residual_record["sft_pred_evidence"] = scored["pred_evidence"]
        residual_record["generated_text"] = generated_text
        residual_record["sft_model_output"] = generated_text
        residual_record["json_valid"] = bool(scored["json_valid"])
        residual_record["parse_errors"] = scored["parse_errors"]
        residual_record["error_type"] = scored["error_type"]
        residual_record["event_f1"] = float(scored["event_f1"])
        residual_record["event_precision"] = float(scored["event_precision"])
        residual_record["event_recall"] = float(scored["event_recall"])
        residual_record["mean_iou"] = float(scored["mean_iou"])
        residual_record["boundary_mae"] = float(scored["boundary_mae"])
        residual_record["matches"] = scored["matches"]
        residual_record["unmatched_pred_indices"] = scored["unmatched_pred_indices"]
        residual_record["unmatched_gt_indices"] = scored["unmatched_gt_indices"]
        residual_record["residual_mining"] = {
            "model_path": args.model_path,
            "tau_match": float(args.tau_match),
            "tau_good": float(args.tau_good),
            "series_length": int(args.series_length),
            "max_index": int(args.max_index),
            "prediction_schema": str(args.prediction_schema),
            "source_input": str(input_path),
        }
        residual_records.append(residual_record)

    _write_jsonl(residual_records, output_path)
    summary = residual_pool_summary(residual_records)
    summary["residual_pool_path"] = str(output_path)
    summary["input_path"] = str(input_path)
    summary["input_kind"] = input_kind
    summary["model_path"] = args.model_path
    summary["tau_match"] = float(args.tau_match)
    summary["tau_good"] = float(args.tau_good)
    summary["series_length"] = int(args.series_length)
    summary["max_index"] = int(args.max_index)
    summary["prediction_schema"] = str(args.prediction_schema)
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else output_path.with_suffix(".summary.json")
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build residual_pool.jsonl from SFT inference on the real train split.")
    parser.add_argument("--model-path", required=True, help="Two-stage SFT model path.")
    parser.add_argument("--real-train-data", required=True, help="Original real train.json or prepared train.jsonl.")
    parser.add_argument("--output", required=True, help="Output residual_pool.jsonl path.")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--config", default=None, help="Optional training config used to rebuild prompts from raw train.json.")
    parser.add_argument("--source-root", default=None, help="Dataset source root for raw train.json input.")
    parser.add_argument("--record-output-root", default=None, help="Cache root for rebuilt records/images when raw train.json is used.")
    parser.add_argument("--image-root", default=None, help="Image root for relative image paths in prepared JSONL input.")
    parser.add_argument("--split", default=None, help="Split name for raw JSON input. Defaults to input filename stem.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tau-match", type=float, default=DEFAULT_TAU_MATCH)
    parser.add_argument("--tau-good", type=float, default=DEFAULT_TAU_GOOD)
    parser.add_argument("--series-length", type=int, default=DEFAULT_SERIES_LENGTH)
    parser.add_argument("--max-index", type=int, default=DEFAULT_MAX_INDEX)
    parser.add_argument(
        "--prediction-schema",
        default=DEFAULT_PREDICTION_SCHEMA,
        choices=["events", "evidence", "auto"],
    )
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.tau_match < 0 or args.tau_match > 1:
        raise ValueError("--tau-match must be in [0, 1]")
    if args.tau_good < 0 or args.tau_good > 1:
        raise ValueError("--tau-good must be in [0, 1]")
    if args.series_length <= 0:
        raise ValueError("--series-length must be positive")
    if args.max_index < 0:
        raise ValueError("--max-index must be non-negative")
    build_residual_pool(args)


if __name__ == "__main__":
    main()
