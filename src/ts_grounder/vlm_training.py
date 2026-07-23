from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .utils import dump_json, ensure_dir, get_world_size, is_main_process
from .vlm_eval import evaluate_predictions, load_jsonl
from .vlm_prompting import build_chat_messages


def _require_training_deps() -> None:
    missing = []
    for module_name in ("datasets", "transformers", "accelerate"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            f"缺少训练依赖：{joined}。请先在目标环境安装 requirements.txt，再把 training.mode 设为 train。"
        )


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


def _decode_new_tokens(processor, model_inputs, generated_ids) -> str:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return _clean_generated_text(decoded)

    prompt_length = int(input_ids.shape[1])
    new_token_ids = generated_ids[:, prompt_length:]
    if new_token_ids.shape[1] == 0:
        return ""

    decoded = processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    return _clean_generated_text(decoded)


def _record_task_type(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("task_type", "grounding"))


def _filter_records_for_generation_eval(
    records: list[dict[str, Any]],
    *,
    task_types: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    selected = records
    if task_types is not None:
        allowed = {str(item) for item in task_types}
        selected = [record for record in selected if _record_task_type(record) in allowed]
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    return selected


def _normalize_string_list(value: Any, *, default: list[str]) -> list[str]:
    if value in (None, "", []):
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _infer_model_device(model) -> Any:
    import torch

    return next(
        (param.device for param in model.parameters() if param.device.type != "meta"),
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )


def _generate_predictions_for_records(
    *,
    processor,
    model,
    records: list[dict[str, Any]],
    output_path: str | Path,
    max_new_tokens: int,
    device,
) -> dict[str, Any]:
    import torch

    predictions = []
    model.eval()
    for record in records:
        from PIL import Image

        image = Image.open(record["image_path"]).convert("RGB")
        messages = build_chat_messages(
            system_prompt=record["system_prompt"],
            user_prompt=record["user_prompt"],
        )
        if not hasattr(processor, "apply_chat_template"):
            raise ValueError("当前 processor 不支持 apply_chat_template，无法走通用推理流程。")
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_text = _decode_new_tokens(processor, model_inputs, outputs)
        predictions.append({"id": record["id"], "generated_text": generated_text})

    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return evaluate_predictions(records, predictions)


@dataclass
class PreparedDataset:
    root: Path
    manifest: dict[str, Any]

    @property
    def train_jsonl(self) -> Path:
        return self.root / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.root / "val.jsonl"

    @property
    def test_jsonl(self) -> Path:
        return self.root / "test.jsonl"


def load_prepared_dataset(dataset_root: str | Path) -> PreparedDataset:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PreparedDataset(root=dataset_root, manifest=manifest)


class JsonlVisionSFTDataset:
    def __init__(self, jsonl_path: str | Path) -> None:
        self.records = load_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class VisionChatCollator:
    def __init__(
        self,
        processor,
        processor_kwargs: dict[str, Any] | None = None,
        evidence_loss_weight: float = 0.0,
        summary_semantic_loss_weight: float = 0.0,
    ) -> None:
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        self.evidence_loss_weight = float(evidence_loss_weight)
        self.summary_semantic_loss_weight = float(summary_semantic_loss_weight)
        if self.evidence_loss_weight > 0 or self.summary_semantic_loss_weight > 0:
            try:
                self.processor.tokenizer("", add_special_tokens=False, return_offsets_mapping=True)
            except Exception as exc:
                raise ValueError(
                    "启用 evidence/summary semantic loss 时，tokenizer 必须支持 return_offsets_mapping。"
                ) from exc

    @staticmethod
    def _find_evidence_char_span(assistant_text: str) -> tuple[int, int]:
        evidence_key = '"evidence"'
        summary_key = '"summary":'
        evidence_start = assistant_text.find(evidence_key)
        summary_start = assistant_text.find(summary_key)
        if evidence_start < 0 or summary_start < 0:
            raise ValueError(f"assistant_text 中未找到 evidence/summary 字段：{assistant_text}")
        if evidence_start > 0 and assistant_text[evidence_start - 1] == "{":
            evidence_start -= 1
        evidence_end = summary_start
        if summary_start > 0 and assistant_text[summary_start - 1] == ",":
            evidence_end = summary_start - 1
        if evidence_end <= evidence_start:
            raise ValueError(f"assistant_text 中 evidence span 非法：{assistant_text}")
        return evidence_start, evidence_end

    @staticmethod
    def _find_summary_char_span(assistant_text: str) -> tuple[int, int]:
        summary_key = '"summary":'
        key_pos = assistant_text.find(summary_key)
        if key_pos < 0:
            raise ValueError(f"assistant_text 中未找到 summary 字段：{assistant_text}")
        cursor = key_pos + len(summary_key)
        while cursor < len(assistant_text) and assistant_text[cursor].isspace():
            cursor += 1
        if cursor >= len(assistant_text) or assistant_text[cursor] != '"':
            raise ValueError(f"assistant_text 中 summary 起始引号非法：{assistant_text}")
        cursor += 1
        start = cursor
        escaped = False
        while cursor < len(assistant_text):
            ch = assistant_text[cursor]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                end = cursor
                if end <= start:
                    raise ValueError(f"assistant_text 中 summary span 非法：{assistant_text}")
                return start, end
            cursor += 1
        raise ValueError(f"assistant_text 中 summary 结束引号缺失：{assistant_text}")

    def _build_field_mask(
        self,
        *,
        prompt_text: str,
        full_text: str,
        assistant_text: str,
        prompt_length: int,
        full_seq_len: int,
        field_span: tuple[int, int],
    ):
        import torch

        tokenizer = self.processor.tokenizer
        assistant_start = full_text.rfind(assistant_text)
        if assistant_start < 0:
            raise ValueError("无法在 full_text 中定位 assistant_text，无法构造字段 mask。")

        field_start_local, field_end_local = field_span
        field_start_global = assistant_start + field_start_local
        field_end_global = assistant_start + field_end_local

        prompt_token_count = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        full_encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = full_encoding["offset_mapping"]

        field_mask = torch.zeros(full_seq_len, dtype=torch.bool)
        for token_idx, (char_start, _char_end) in enumerate(offsets):
            if char_start < field_start_global or char_start >= field_end_global:
                continue
            if token_idx < prompt_token_count:
                continue
            seq_pos = int(prompt_length) + (token_idx - prompt_token_count)
            if 0 <= seq_pos < full_seq_len:
                field_mask[seq_pos] = True
        return field_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image

        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("当前 processor 不支持 apply_chat_template，无法走通用 chat-format VLM SFT。")

        images = [Image.open(record["image_path"]).convert("RGB") for record in batch]
        prompt_texts = []
        full_texts = []
        for record in batch:
            prompt_messages = build_chat_messages(
                system_prompt=record["system_prompt"],
                user_prompt=record["user_prompt"],
            )
            full_messages = build_chat_messages(
                system_prompt=record["system_prompt"],
                user_prompt=record["user_prompt"],
                assistant_text=record["assistant_text"],
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        processor_call_kwargs = {
            "padding": True,
            "return_tensors": "pt",
            **self.processor_kwargs,
        }
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            **processor_call_kwargs,
        )
        full_batch = self.processor(
            text=full_texts,
            images=images,
            **processor_call_kwargs,
        )

        labels = full_batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        pad_token_id = self.processor.tokenizer.pad_token_id
        for idx, prompt_length in enumerate(prompt_lengths):
            labels[idx, : int(prompt_length)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        full_batch["labels"] = labels
        if self.evidence_loss_weight > 0:
            import torch

            evidence_masks = []
            for idx, record in enumerate(batch):
                if record.get("metadata", {}).get("task_type") == "qa":
                    evidence_mask = torch.zeros(int(full_batch["input_ids"].shape[1]), dtype=torch.bool)
                else:
                    evidence_mask = self._build_field_mask(
                        prompt_text=prompt_texts[idx],
                        full_text=full_texts[idx],
                        assistant_text=record["assistant_text"],
                        prompt_length=int(prompt_lengths[idx]),
                        full_seq_len=int(full_batch["input_ids"].shape[1]),
                        field_span=self._find_evidence_char_span(record["assistant_text"]),
                    )
                evidence_mask &= labels[idx] != -100
                evidence_masks.append(evidence_mask)
            full_batch["evidence_mask"] = torch.stack(evidence_masks, dim=0)

        if self.summary_semantic_loss_weight > 0:
            import torch

            summary_masks = []
            summary_texts = []
            for idx, record in enumerate(batch):
                if record.get("metadata", {}).get("task_type") == "qa":
                    summary_value = str(record.get("target", {}).get("answer", record["assistant_text"]))
                    summary_mask = torch.zeros(int(full_batch["input_ids"].shape[1]), dtype=torch.bool)
                    summary_texts.append(summary_value)
                    summary_masks.append(summary_mask)
                    continue
                try:
                    summary_value = json.loads(record["assistant_text"]).get("summary", "")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"assistant_text 不是合法 JSON：{record['assistant_text']}") from exc
                summary_texts.append(summary_value)
                summary_mask = self._build_field_mask(
                    prompt_text=prompt_texts[idx],
                    full_text=full_texts[idx],
                    assistant_text=record["assistant_text"],
                    prompt_length=int(prompt_lengths[idx]),
                    full_seq_len=int(full_batch["input_ids"].shape[1]),
                    field_span=self._find_summary_char_span(record["assistant_text"]),
                )
                summary_mask &= labels[idx] != -100
                summary_masks.append(summary_mask)
            full_batch["summary_mask"] = torch.stack(summary_masks, dim=0)
            full_batch["summary_texts"] = summary_texts
        return full_batch


def prepare_vlm_dataset(cfg: dict[str, Any], repo_root: Path, output_dir: Path) -> PreparedDataset:
    from .vlm_data import build_vlm_sft_dataset, resolve_source_root

    data_cfg = cfg["data"]
    dataset_root = output_dir / "dataset_cache"
    source_root = resolve_source_root(data_cfg.get("source_root"), repo_root=repo_root)
    manifest = build_vlm_sft_dataset(
        source_root=source_root,
        output_root=dataset_root,
        split_file_suffix=str(data_cfg.get("split_file_suffix", "")),
        image_subdir=data_cfg.get("image_subdir"),
        include_grounding_records=bool(data_cfg.get("include_grounding_records", True)),
        include_qa_pairs=bool(data_cfg.get("include_qa_pairs", False)),
        qa_system_prompt=data_cfg.get("qa_system_prompt"),
        qa_user_prompt_template=str(
            data_cfg.get(
                "qa_user_prompt_template",
                "Answer this question about the time-series plot exactly and concisely.\n\nQuestion: {question}",
            )
        ),
        include_indexed_series_text=bool(data_cfg.get("include_indexed_series_text", False)),
        indexed_series_precision=int(data_cfg.get("indexed_series_precision", 4)),
        indexed_series_compact=bool(data_cfg.get("indexed_series_compact", False)),
        max_train_samples=data_cfg.get("max_train_samples"),
        max_val_samples=data_cfg.get("max_val_samples"),
        max_test_samples=data_cfg.get("max_test_samples"),
        balanced_by_type=bool(data_cfg.get("balanced_by_type", True)),
        source_dataset_filter=data_cfg.get("source_dataset_filter"),
        series_normalization=data_cfg.get("series_normalization"),
        seed=int(cfg.get("seed", 2026)),
        system_prompt=cfg["prompt"]["system_prompt"],
        user_prompt=cfg["prompt"]["user_prompt"],
    )
    return PreparedDataset(root=dataset_root, manifest=manifest)


def run_prepare_only(prepared: PreparedDataset, output_dir: Path) -> dict[str, Any]:
    val_records = load_jsonl(prepared.val_jsonl)
    oracle_predictions = []
    for record in val_records:
        oracle_predictions.append({"id": record["id"], "generated_text": record["assistant_text"]})
    oracle_metrics = evaluate_predictions(val_records, oracle_predictions)
    summary = {
        "mode": "prepare_only",
        "dataset_root": str(prepared.root),
        "manifest": prepared.manifest,
        "oracle_metrics": oracle_metrics,
    }
    dump_json(summary, output_dir / "prepare_summary.json")
    return summary


def _resolve_resume_checkpoint(
    resume_value: Any,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> str | None:
    if resume_value in (None, False, ""):
        return None
    if resume_value is True:
        resume_value = "latest"

    resume_text = str(resume_value).strip()
    if resume_text.lower() in {"latest", "last"}:
        candidates = []
        for child in checkpoint_dir.glob("checkpoint-*"):
            if not child.is_dir():
                continue
            trainer_state = child / "trainer_state.json"
            if not trainer_state.exists():
                continue
            try:
                step = int(child.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            candidates.append((step, child))
        if not candidates:
            raise ValueError(f"未在 {checkpoint_dir} 下找到可恢复的完整 checkpoint。")
        return str(max(candidates, key=lambda item: item[0])[1])

    resume_path = Path(resume_text).expanduser()
    if not resume_path.is_absolute():
        repo_candidate = Path.cwd() / resume_path
        output_candidate = output_dir / resume_path
        checkpoint_candidate = checkpoint_dir / resume_path
        if repo_candidate.exists():
            resume_path = repo_candidate
        elif output_candidate.exists():
            resume_path = output_candidate
        elif checkpoint_candidate.exists():
            resume_path = checkpoint_candidate

    if not resume_path.exists():
        raise ValueError(f"resume_from_checkpoint 指向的路径不存在：{resume_path}")
    return str(resume_path)


def run_hf_vlm_sft(cfg: dict[str, Any], prepared: PreparedDataset, output_dir: Path) -> dict[str, Any]:
    _require_training_deps()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model_name_or_path = model_cfg.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("training.mode=train 时必须提供 model.model_name_or_path。")

    fsdp = train_cfg.get("fsdp")
    fsdp_config = dict(train_cfg.get("fsdp_config", {}) or {})
    if fsdp and get_world_size() < 2:
        raise ValueError("当前配置启用了 FSDP，但训练进程数小于 2。请使用 torchrun 在多卡模式下启动。")

    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    if fsdp and fsdp_config.get("activation_checkpointing") and gradient_checkpointing:
        if is_main_process():
            print("FSDP activation_checkpointing 已启用，关闭原生 gradient_checkpointing 以避免重复开销。")
        gradient_checkpointing = False

    processor_kwargs = dict(model_cfg.get("processor_kwargs", {}) or {})
    model_init_kwargs = dict(model_cfg.get("from_pretrained_kwargs", {}) or {})
    torch_dtype = model_init_kwargs.get("torch_dtype")
    if isinstance(torch_dtype, str):
        normalized = torch_dtype.strip().lower()
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
            raise ValueError(f"Unsupported torch_dtype value: {torch_dtype}")
        model_init_kwargs["torch_dtype"] = dtype_map[normalized]

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        **processor_kwargs,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        **model_init_kwargs,
    )
    if gradient_checkpointing and hasattr(model, "config") and getattr(model.config, "use_cache", None):
        model.config.use_cache = False

    train_dataset = JsonlVisionSFTDataset(prepared.train_jsonl)
    val_dataset = JsonlVisionSFTDataset(prepared.val_jsonl)
    evidence_loss_weight = float(train_cfg.get("evidence_loss_weight", 0.0))
    summary_semantic_loss_weight = float(train_cfg.get("summary_semantic_loss_weight", 0.0))
    summary_semantic_model = str(
        train_cfg.get("summary_semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    summary_semantic_max_length = int(train_cfg.get("summary_semantic_max_length", 128))

    collator = VisionChatCollator(
        processor=processor,
        processor_kwargs=processor_kwargs,
        evidence_loss_weight=evidence_loss_weight,
        summary_semantic_loss_weight=summary_semantic_loss_weight,
    )
    checkpoint_output_dir = train_cfg.get("checkpoint_output_dir")
    if checkpoint_output_dir:
        checkpoint_dir = ensure_dir(Path(checkpoint_output_dir).expanduser())
    else:
        checkpoint_dir = ensure_dir(output_dir / "hf_checkpoints")
    load_best_model_at_end = bool(train_cfg.get("load_best_model_at_end", False))
    metric_for_best_model = train_cfg.get("metric_for_best_model")
    if load_best_model_at_end and not metric_for_best_model:
        metric_for_best_model = "eval_loss"
    greater_is_better = train_cfg.get("greater_is_better")
    if greater_is_better is None and metric_for_best_model:
        greater_is_better = not str(metric_for_best_model).endswith("loss")
    save_total_limit = train_cfg.get("save_total_limit")
    resume_from_checkpoint = _resolve_resume_checkpoint(
        train_cfg.get("resume_from_checkpoint"),
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )

    eval_strategy = str(train_cfg.get("evaluation_strategy", "steps"))
    save_strategy = str(train_cfg.get("save_strategy", "steps"))
    optim = train_cfg.get("optim")
    memory_optimized_loss = bool(train_cfg.get("memory_optimized_loss", False))
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 100)),
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        report_to="none",
        remove_unused_columns=False,
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=int(save_total_limit) if save_total_limit is not None else None,
        do_train=True,
        do_eval=eval_strategy != "no",
        fsdp=fsdp,
        fsdp_config=fsdp_config if fsdp else None,
        optim=str(optim) if optim is not None else "adamw_torch",
    )

    summary_semantic_proj_dim = int(train_cfg.get("summary_semantic_proj_dim", 384))

    trainer_cls = Trainer
    if memory_optimized_loss or evidence_loss_weight > 0 or summary_semantic_loss_weight > 0:
        if summary_semantic_loss_weight > 0:
            vocab_config = getattr(model.config, "text_config", model.config)
            hidden_size = int(getattr(vocab_config, "hidden_size", model.config.hidden_size))
            model.summary_semantic_proj = nn.Linear(hidden_size, summary_semantic_proj_dim, bias=False)

        class _LabelOnlyLogitsTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._summary_encoder = None
                self._summary_encoder_device = None

            @staticmethod
            def _compute_token_ce(model, logits, labels, shift_labels, num_items_in_batch=None):
                vocab_config = getattr(model.config, "text_config", model.config)
                return model.loss_function(
                    logits=logits,
                    labels=labels,
                    shift_labels=shift_labels,
                    vocab_size=vocab_config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                )

            def _get_summary_encoder(self, device: torch.device):
                if self._summary_encoder is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise ImportError(
                            "summary_semantic_loss_weight 需要 sentence-transformers。"
                            "请在训练环境中安装 sentence-transformers。"
                        ) from exc
                    encoder = SentenceTransformer(summary_semantic_model)
                    encoder.eval()
                    self._summary_encoder = encoder
                if self._summary_encoder_device != device:
                    self._summary_encoder.to(device)
                    self._summary_encoder_device = device
                return self._summary_encoder

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                if labels is None:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )

                summary_mask = inputs.get("summary_mask")
                summary_texts = inputs.get("summary_texts")
                evidence_mask = inputs.get("evidence_mask")
                shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
                keep_positions = torch.nonzero((shift_labels != -100).any(dim=0), as_tuple=False).squeeze(-1)
                if keep_positions.numel() == 0:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )

                model_inputs = dict(inputs)
                model_inputs.pop("labels", None)
                model_inputs.pop("evidence_mask", None)
                model_inputs.pop("summary_mask", None)
                model_inputs.pop("summary_texts", None)
                if summary_semantic_loss_weight > 0:
                    model_inputs["output_hidden_states"] = True

                if memory_optimized_loss:
                    keep_positions = keep_positions.to(labels.device)
                    outputs = model(**model_inputs, logits_to_keep=keep_positions)
                    selected_shift_labels = shift_labels.index_select(1, keep_positions.to(shift_labels.device))
                    current_loss = self._compute_token_ce(
                        model,
                        outputs.logits,
                        labels,
                        selected_shift_labels,
                        num_items_in_batch=num_items_in_batch,
                    )
                else:
                    outputs = model(**model_inputs, labels=labels)
                    current_loss = outputs.loss

                loss = current_loss
                if evidence_loss_weight > 0:
                    if evidence_mask is None:
                        raise ValueError("evidence_loss_weight > 0 时，batch 中缺少 evidence_mask。")
                    shift_evidence_mask = F.pad(evidence_mask.to(torch.bool), (0, 1), value=False)[..., 1:].contiguous()
                    evidence_shift_labels = shift_labels.masked_fill(~shift_evidence_mask, -100)
                    if (evidence_shift_labels != -100).any():
                        if memory_optimized_loss:
                            evidence_keep_positions = torch.nonzero(
                                (evidence_shift_labels != -100).any(dim=0),
                                as_tuple=False,
                            ).squeeze(-1)
                            evidence_keep_positions = evidence_keep_positions.to(labels.device)
                            evidence_outputs = model(**model_inputs, logits_to_keep=evidence_keep_positions)
                            selected_evidence_shift_labels = evidence_shift_labels.index_select(
                                1, evidence_keep_positions.to(evidence_shift_labels.device)
                            )
                            evidence_loss = self._compute_token_ce(
                                model,
                                evidence_outputs.logits,
                                labels,
                                selected_evidence_shift_labels,
                                num_items_in_batch=num_items_in_batch,
                            )
                        else:
                            evidence_loss = self._compute_token_ce(
                                model,
                                outputs.logits,
                                labels,
                                evidence_shift_labels,
                                num_items_in_batch=num_items_in_batch,
                            )
                        loss = loss + (evidence_loss_weight * evidence_loss)
                if summary_semantic_loss_weight > 0:
                    if summary_mask is None or summary_texts is None:
                        raise ValueError("summary_semantic_loss_weight > 0 时，batch 中缺少 summary_mask/summary_texts。")
                    if not hasattr(model, "summary_semantic_proj"):
                        raise ValueError("summary_semantic_loss_weight > 0 时，model 缺少 summary_semantic_proj。")

                    shift_summary_mask = F.pad(summary_mask.to(torch.bool), (0, 1), value=False)[..., 1:].contiguous()
                    valid_summary = shift_summary_mask.any(dim=1)
                    if valid_summary.any():
                        hidden_states = outputs.hidden_states[-1]
                        mask = shift_summary_mask.unsqueeze(-1).to(hidden_states.device)
                        masked_hidden = hidden_states * mask
                        lengths = mask.sum(dim=1).clamp_min(1)
                        summary_repr = masked_hidden.sum(dim=1) / lengths
                        proj = model.summary_semantic_proj(summary_repr)
                        proj = F.normalize(proj, p=2, dim=-1)

                        encoder = self._get_summary_encoder(proj.device)
                        with torch.no_grad():
                            target = encoder.encode(
                                summary_texts,
                                convert_to_tensor=True,
                                device=proj.device,
                                normalize_embeddings=True,
                                batch_size=len(summary_texts),
                                show_progress_bar=False,
                            )
                        # SentenceTransformer may return inference-mode tensors; clone to make them
                        # compatible with autograd operations in the loss computation.
                        target = target.detach().clone()
                        if proj.shape[-1] != target.shape[-1]:
                            raise ValueError(
                                f"summary_semantic_proj_dim({proj.shape[-1]}) 与 "
                                f"encoder dim({target.shape[-1]}) 不一致。"
                                "请通过 summary_semantic_proj_dim 调整。"
                            )
                        semantic_loss = (1 - (proj * target).sum(dim=-1))
                        semantic_loss = semantic_loss[valid_summary].mean()
                        loss = loss + (summary_semantic_loss_weight * semantic_loss)
                if return_outputs:
                    return loss, outputs
                return loss

        trainer_cls = _LabelOnlyLogitsTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if eval_strategy != "no" else None,
        data_collator=collator,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "model"))
    if hasattr(processor, "save_pretrained") and trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir / "model"))

    summary = {
        "mode": "train",
        "dataset_root": str(prepared.root),
        "manifest": prepared.manifest,
        "checkpoint_dir": str(checkpoint_dir),
        "resumed_from_checkpoint": resume_from_checkpoint,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": getattr(train_result, "metrics", {}),
    }
    if bool(train_cfg.get("generation_eval_after_train", False)):
        if get_world_size() > 1:
            if trainer.is_world_process_zero():
                summary["generation_eval_skipped"] = "generation_eval_after_train is only supported for single-process runs."
        elif trainer.is_world_process_zero():
            eval_splits = _normalize_string_list(train_cfg.get("generation_eval_splits"), default=["val"])
            task_types_value = train_cfg.get("generation_eval_task_types")
            eval_task_types = (
                None
                if task_types_value in (None, "", [])
                else _normalize_string_list(task_types_value, default=[])
            )
            eval_max_samples = train_cfg.get("generation_eval_max_samples")
            eval_max_new_tokens = int(train_cfg.get("generation_eval_max_new_tokens", 64))
            eval_device = _infer_model_device(model)
            generation_eval: dict[str, Any] = {}
            for split in eval_splits:
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"Unsupported generation_eval split: {split}")
                dataset_jsonl = getattr(prepared, f"{split}_jsonl")
                records = load_jsonl(dataset_jsonl)
                records = _filter_records_for_generation_eval(
                    records,
                    task_types=eval_task_types,
                    max_samples=eval_max_samples,
                )
                task_suffix = "all" if eval_task_types is None else "_".join(eval_task_types)
                predictions_path = output_dir / "eval" / f"{split}_{task_suffix}_predictions.jsonl"
                metrics = _generate_predictions_for_records(
                    processor=processor,
                    model=model,
                    records=records,
                    output_path=predictions_path,
                    max_new_tokens=eval_max_new_tokens,
                    device=eval_device,
                )
                metrics_path = predictions_path.with_suffix(".metrics.json")
                dump_json(metrics, metrics_path)
                generation_eval[split] = {
                    "dataset_jsonl": str(dataset_jsonl),
                    "predictions_path": str(predictions_path),
                    "metrics_path": str(metrics_path),
                    "task_types": eval_task_types,
                    "max_samples": eval_max_samples,
                    "max_new_tokens": eval_max_new_tokens,
                    "metrics": metrics,
                }
            summary["generation_eval"] = generation_eval
    if trainer.is_world_process_zero():
        dump_json(summary, output_dir / "train_summary.json")
    return summary


def generate_predictions(
    *,
    model_path: str,
    dataset_jsonl: str | Path,
    output_path: str | Path,
    max_new_tokens: int = 256,
    task_types: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    _require_training_deps()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    def _resolve_predict_dtype(value: str | None):
        if value in (None, ""):
            return None
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
            raise ValueError(f"Unsupported TS_GROUNDER_PREDICT_TORCH_DTYPE value: {value}")
        return dtype_map[normalized]

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {}
    predict_dtype = _resolve_predict_dtype(os.environ.get("TS_GROUNDER_PREDICT_TORCH_DTYPE"))
    if predict_dtype is not None:
        model_kwargs["torch_dtype"] = predict_dtype
    device_map = os.environ.get("TS_GROUNDER_PREDICT_DEVICE_MAP")
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
    records = load_jsonl(dataset_jsonl)
    records = _filter_records_for_generation_eval(
        records,
        task_types=task_types,
        max_samples=max_samples,
    )
    return _generate_predictions_for_records(
        processor=processor,
        model=model,
        records=records,
        output_path=output_path,
        max_new_tokens=max_new_tokens,
        device=device,
    )
