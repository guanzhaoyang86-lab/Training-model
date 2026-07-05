from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ts_grounder.event_metrics import (
        DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS,
        DEFAULT_MAX_INDEX,
        DEFAULT_PREDICTION_SCHEMA,
        DEFAULT_RESIDUAL_SAMPLING_RATIOS,
        DEFAULT_SERIES_LENGTH,
        DEFAULT_TAU_GOOD,
        DEFAULT_TAU_MATCH,
        build_balanced_residual_epoch,
        parse_sampling_ratios,
    )
    from ts_grounder.rl_reward import (
        DEFAULT_REWARD_WEIGHTS,
        compute_boundary_aware_grounder_reward_details,
        compute_grounder_reward_details,
        resolve_boundary_aware_reward_weights,
        resolve_reward_weights,
    )
    from ts_grounder.vlm_eval import load_jsonl
    from ts_grounder.vlm_prompting import build_chat_messages
    from ts_grounder.vlm_training import _clean_generated_text, _generate_predictions_for_records
    from ts_grounder.utils import load_yaml
else:
    from .event_metrics import (
        DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS,
        DEFAULT_MAX_INDEX,
        DEFAULT_PREDICTION_SCHEMA,
        DEFAULT_RESIDUAL_SAMPLING_RATIOS,
        DEFAULT_SERIES_LENGTH,
        DEFAULT_TAU_GOOD,
        DEFAULT_TAU_MATCH,
        build_balanced_residual_epoch,
        parse_sampling_ratios,
    )
    from .rl_reward import (
        DEFAULT_REWARD_WEIGHTS,
        compute_boundary_aware_grounder_reward_details,
        compute_grounder_reward_details,
        resolve_boundary_aware_reward_weights,
        resolve_reward_weights,
    )
    from .vlm_eval import load_jsonl
    from .vlm_prompting import build_chat_messages
    from .vlm_training import _clean_generated_text, _generate_predictions_for_records
    from .utils import load_yaml


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


def _model_device(model):
    import torch

    return next(
        (param.device for param in model.parameters() if param.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _move_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


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


def _resolve_image_path(record: dict[str, Any], image_root: str | None) -> Path:
    raw_path = record.get("image_path") or record.get("image_relpath")
    if raw_path is None:
        raise KeyError(f"record {record.get('id', '<unknown>')} is missing image_path")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if image_root:
        return Path(image_root).expanduser() / path
    return path


def _seq_len_for_record(record: dict[str, Any], ground_truth: Any) -> int:
    for key in ("seq_len", "series_length"):
        if key in record:
            return int(record[key])
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("series_length") is not None:
        return int(metadata["series_length"])
    if isinstance(ground_truth, dict):
        for key in ("point_labels", "labels"):
            labels = ground_truth.get(key)
            if isinstance(labels, list):
                return len(labels)
    if isinstance(ground_truth, list):
        return len(ground_truth)
    raise KeyError(f"record {record.get('id', '<unknown>')} is missing seq_len")


def _ground_truth_for_record(record: dict[str, Any]) -> Any:
    for key in ("ground_truth", "target"):
        if key in record:
            return record[key]
    raw_sample = record.get("raw_sample")
    if isinstance(raw_sample, dict):
        if raw_sample.get("events") is not None:
            return {"evidence": raw_sample.get("events", [])}
        if raw_sample.get("point_labels") is not None:
            return raw_sample["point_labels"]
    for key in ("point_labels", "labels"):
        if key in record:
            return record[key]
    raise KeyError(f"record {record.get('id', '<unknown>')} is missing ground truth")


def _generate_candidates(
    *,
    processor,
    model,
    image,
    system_prompt: str,
    user_prompt: str,
    device,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    messages = build_chat_messages(system_prompt=system_prompt, user_prompt=user_prompt)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    model_inputs = _move_to_device(model_inputs, device)

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_return_sequences": int(num_generations),
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)
    return _decode_new_tokens(processor, model_inputs, generated_ids)


def _candidate_log_probs(
    *,
    processor,
    model,
    image,
    system_prompt: str,
    user_prompt: str,
    candidate_texts: list[str],
    device,
):
    import torch

    prompt_messages = build_chat_messages(system_prompt=system_prompt, user_prompt=user_prompt)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_texts = []
    for candidate in candidate_texts:
        full_messages = build_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_text=candidate,
        )
        full_texts.append(
            processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    prompt_batch = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    prompt_length = int(prompt_batch["attention_mask"].sum(dim=1)[0].item())
    full_batch = processor(
        text=full_texts,
        images=[image for _ in candidate_texts],
        padding=True,
        return_tensors="pt",
    )
    full_batch = _move_to_device(full_batch, device)
    input_ids = full_batch["input_ids"]
    attention_mask = full_batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    outputs = model(**full_batch)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    target_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = target_logits - torch.logsumexp(shift_logits, dim=-1)

    token_positions = torch.arange(shift_labels.shape[1], device=shift_labels.device).unsqueeze(0)
    response_mask = token_positions >= max(0, prompt_length - 1)
    response_mask = response_mask & attention_mask[:, 1:].to(torch.bool)
    lengths = response_mask.sum(dim=1).clamp_min(1)
    return (token_log_probs * response_mask).sum(dim=1) / lengths


def _normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std_reward = math.sqrt(variance)
    return [(reward - mean_reward) / (std_reward + 1e-6) for reward in rewards]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_trainer_state(checkpoint_dir: str | Path | None) -> dict[str, Any]:
    if not checkpoint_dir:
        return {}
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    model,
    processor,
    optimizer,
    state: dict[str, Any],
    save_optimizer_state: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if save_optimizer_state:
        import torch

        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")


def _maybe_resume_optimizer(optimizer, checkpoint_dir: str | Path | None) -> bool:
    if not checkpoint_dir:
        return False
    optimizer_path = Path(checkpoint_dir) / "optimizer.pt"
    if not optimizer_path.exists():
        return False
    import torch

    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    return True


def _build_optimizer(model, args):
    import torch
    from transformers import Adafactor

    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=float(args.learning_rate),
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _reward_weights_from_args(args) -> dict[str, float]:
    return resolve_reward_weights(
        {
            "event_f1": float(args.reward_event_f1_weight),
            "boundary_iou": float(args.reward_boundary_iou_weight),
            "hallucination_penalty": float(args.reward_hallucination_penalty_weight),
        }
    )


def _boundary_reward_weights_from_args(args) -> dict[str, float]:
    return resolve_boundary_aware_reward_weights(
        {
            "point": float(args.reward_point_weight),
            "event": float(args.reward_event_weight),
            "iou": float(args.reward_iou_weight),
            "boundary": float(args.reward_boundary_weight),
            "type": float(args.reward_type_weight),
        }
    )


def _next_position(epoch: int, record_index: int, num_records: int) -> tuple[int, int]:
    next_record_index = record_index + 1
    next_epoch = epoch
    if next_record_index >= num_records:
        next_epoch += 1
        next_record_index = 0
    return next_epoch, next_record_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal GRPO-style RL training entry for TS Grounder.")
    parser.add_argument("--config", type=str, default=None, help="Optional RL config YAML/JSON; CLI args override it.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--reference_model_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--residual_pool_file", type=str, default=None)
    parser.add_argument("--use_residual_pool", action="store_true")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--reward_event_f1_weight", type=float, default=DEFAULT_REWARD_WEIGHTS["event_f1"])
    parser.add_argument("--reward_boundary_iou_weight", type=float, default=DEFAULT_REWARD_WEIGHTS["boundary_iou"])
    parser.add_argument(
        "--reward_hallucination_penalty_weight",
        type=float,
        default=DEFAULT_REWARD_WEIGHTS["hallucination_penalty"],
    )
    parser.add_argument("--use_boundary_aware_reward", action="store_true")
    parser.add_argument("--tau_match", type=float, default=DEFAULT_TAU_MATCH)
    parser.add_argument("--tau_good", type=float, default=DEFAULT_TAU_GOOD)
    parser.add_argument("--series_length", type=int, default=DEFAULT_SERIES_LENGTH)
    parser.add_argument("--max_index", type=int, default=DEFAULT_MAX_INDEX)
    parser.add_argument(
        "--prediction_schema",
        type=str,
        default=DEFAULT_PREDICTION_SCHEMA,
        choices=["events", "evidence", "auto"],
    )
    parser.add_argument(
        "--reward_point_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["point"],
    )
    parser.add_argument(
        "--reward_event_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["event"],
    )
    parser.add_argument(
        "--reward_iou_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["iou"],
    )
    parser.add_argument(
        "--reward_boundary_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["boundary"],
    )
    parser.add_argument(
        "--reward_type_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["type"],
    )
    parser.add_argument(
        "--residual_pool_sampling_ratios",
        type=str,
        default=",".join(f"{key}={value}" for key, value in DEFAULT_RESIDUAL_SAMPLING_RATIOS.items()),
    )
    parser.add_argument("--residual_pool_epoch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adafactor", "adamw"])
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    return parser


def _config_defaults_from_file(config_path: str | Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"RL config must be an object: {path}")
    source = cfg.get("rl", cfg.get("grpo", cfg))
    if not isinstance(source, dict):
        raise ValueError(f"RL config must contain an object at top-level, rl, or grpo: {path}")

    known_dests = {action.dest for action in parser._actions}
    defaults: dict[str, Any] = {"config": str(path)}
    aliases = {
        "residual_pool_path": "residual_pool_file",
        "residual_pool": "residual_pool_file",
        "error_type_sampling_ratios": "residual_pool_sampling_ratios",
        "sampling_ratios": "residual_pool_sampling_ratios",
    }
    for key, value in source.items():
        dest = aliases.get(str(key), str(key))
        if dest == "reward_weights" and isinstance(value, dict):
            weight_aliases = {
                "event": "reward_event_weight",
                "event_f1": "reward_event_weight",
                "point": "reward_point_weight",
                "point_f1": "reward_point_weight",
                "iou": "reward_iou_weight",
                "mean_iou": "reward_iou_weight",
                "boundary": "reward_boundary_weight",
                "type": "reward_type_weight",
            }
            for weight_key, weight_value in value.items():
                weight_dest = weight_aliases.get(str(weight_key), str(weight_key))
                if weight_dest in known_dests:
                    defaults[weight_dest] = weight_value
            continue
        if dest in known_dests:
            defaults[dest] = value
    return defaults


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**_config_defaults_from_file(config_args.config, parser))
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    for required_name in ("model_name_or_path", "train_file", "output_dir"):
        if getattr(args, required_name) in (None, ""):
            raise ValueError(f"--{required_name} is required, either on the CLI or in --config")
    if args.use_residual_pool and not args.residual_pool_file:
        raise ValueError("--residual_pool_file is required when --use_residual_pool is enabled")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for group normalization")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive when sampling multiple generations")
    if args.kl_coef < 0:
        raise ValueError("--kl_coef must be non-negative")
    if args.save_steps < 0:
        raise ValueError("--save_steps must be non-negative")
    if args.tau_match < 0 or args.tau_match > 1:
        raise ValueError("--tau_match must be in [0, 1]")
    if args.tau_good < 0 or args.tau_good > 1:
        raise ValueError("--tau_good must be in [0, 1]")
    if args.series_length <= 0:
        raise ValueError("--series_length must be positive")
    if args.max_index < 0:
        raise ValueError("--max_index must be non-negative")
    reward_weights = (
        _boundary_reward_weights_from_args(args)
        if args.use_boundary_aware_reward
        else _reward_weights_from_args(args)
    )
    residual_sampling_ratios = parse_sampling_ratios(args.residual_pool_sampling_ratios)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "rl_rollouts.jsonl"

    model_load_path = args.resume_from_checkpoint or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(model_load_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {}
    torch_dtype = _resolve_dtype(args.torch_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForImageTextToText.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    gradient_checkpointing = not bool(args.disable_gradient_checkpointing)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = _build_optimizer(model, args)
    optimizer_resumed = _maybe_resume_optimizer(optimizer, args.resume_from_checkpoint)

    reference_model = None
    reference_model_path = args.reference_model_path or args.model_name_or_path
    if args.kl_coef > 0:
        reference_model = AutoModelForImageTextToText.from_pretrained(
            reference_model_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        if hasattr(reference_model.config, "use_cache"):
            reference_model.config.use_cache = False
        reference_model = reference_model.to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    train_file = args.residual_pool_file if args.use_residual_pool else args.train_file
    records = load_jsonl(train_file)
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]
    residual_pool_epoch_size = (
        int(args.residual_pool_epoch_size)
        if args.residual_pool_epoch_size is not None
        else len(records)
    )
    num_epochs = max(1, int(math.ceil(float(args.num_train_epochs))))
    trainer_state = _load_trainer_state(args.resume_from_checkpoint)
    global_step = int(trainer_state.get("global_step", 0))
    start_epoch = int(trainer_state.get("next_epoch", 0))
    start_record_index = int(trainer_state.get("next_record_index", 0))
    running_rewards: list[float] = []

    print(
        "RL config:",
        {
            "kl_coef": float(args.kl_coef),
            "config": args.config,
            "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
            "reward_weights": reward_weights,
            "tau_match": float(args.tau_match),
            "tau_good": float(args.tau_good),
            "series_length": int(args.series_length),
            "max_index": int(args.max_index),
            "prediction_schema": str(args.prediction_schema),
            "use_residual_pool": bool(args.use_residual_pool),
            "residual_pool_file": args.residual_pool_file,
            "residual_pool_epoch_size": residual_pool_epoch_size if args.use_residual_pool else None,
            "residual_pool_sampling_ratios": residual_sampling_ratios if args.use_residual_pool else None,
            "reference_model_path": reference_model_path if reference_model is not None else None,
            "optimizer": str(args.optimizer),
            "gradient_checkpointing": gradient_checkpointing,
            "save_steps": int(args.save_steps),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "optimizer_resumed": optimizer_resumed,
        },
    )

    for epoch in range(start_epoch, num_epochs):
        epoch_records = (
            build_balanced_residual_epoch(
                records,
                epoch_size=residual_pool_epoch_size,
                sampling_ratios=residual_sampling_ratios,
                seed=int(args.seed) + epoch,
            )
            if args.use_residual_pool
            else records
        )
        epoch_start_index = start_record_index if epoch == start_epoch else 0
        for record_index in range(epoch_start_index, len(epoch_records)):
            record = epoch_records[record_index]
            image_path = _resolve_image_path(record, args.image_root)
            image = Image.open(image_path).convert("RGB")
            system_prompt = str(record["system_prompt"])
            user_prompt = str(record["user_prompt"])
            ground_truth = _ground_truth_for_record(record)
            seq_len = _seq_len_for_record(record, ground_truth)

            candidates = _generate_candidates(
                processor=processor,
                model=model,
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                device=device,
                num_generations=int(args.num_generations),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            if args.use_boundary_aware_reward:
                reward_details = [
                    compute_boundary_aware_grounder_reward_details(
                        candidate,
                        ground_truth,
                        int(args.series_length),
                        tau_match=float(args.tau_match),
                        max_index=int(args.max_index),
                        reward_weights=reward_weights,
                        prediction_schema=str(args.prediction_schema),
                    )
                    for candidate in candidates
                ]
            else:
                reward_details = [
                    compute_grounder_reward_details(
                        candidate,
                        ground_truth,
                        seq_len,
                        reward_weights=reward_weights,
                    )
                    for candidate in candidates
                ]
            rewards = [float(item["reward"]) for item in reward_details]
            advantages = _normalize_advantages(rewards)

            ref_log_probs = None
            if reference_model is not None:
                with torch.no_grad():
                    ref_log_probs = _candidate_log_probs(
                        processor=processor,
                        model=reference_model,
                        image=image,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        candidate_texts=candidates,
                        device=device,
                    )

            model.train()
            log_probs = _candidate_log_probs(
                processor=processor,
                model=model,
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                candidate_texts=candidates,
                device=device,
            )
            advantage_tensor = torch.tensor(advantages, device=log_probs.device, dtype=log_probs.dtype)
            policy_loss = -(advantage_tensor.detach() * log_probs).mean()
            kl_penalty = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
            sampled_kl = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
            if ref_log_probs is not None:
                ref_log_probs = ref_log_probs.to(device=log_probs.device, dtype=log_probs.dtype)
                log_ratio = log_probs - ref_log_probs
                sampled_kl = log_ratio.mean()
                kl_penalty = 0.5 * log_ratio.pow(2).mean()
            loss = policy_loss + (float(args.kl_coef) * kl_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_rewards.extend(rewards)
            next_epoch, next_record_index = _next_position(epoch, record_index, len(epoch_records))
            _append_jsonl(
                rollout_path,
                {
                    "epoch": epoch,
                    "record_index": record_index,
                    "sample_id": record.get("id") or record.get("sample_id"),
                    "candidates": candidates,
                    "rewards": rewards,
                    "advantages": advantages,
                    "reward_details": reward_details,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "kl_penalty": float(kl_penalty.detach().cpu()),
                    "sampled_kl": float(sampled_kl.detach().cpu()),
                    "ref_log_probs": (
                        None
                        if ref_log_probs is None
                        else [float(value) for value in ref_log_probs.detach().cpu().tolist()]
                    ),
                    "next_epoch": next_epoch,
                    "next_record_index": next_record_index,
                },
            )

            if global_step % int(args.logging_steps) == 0:
                recent = running_rewards[-int(args.logging_steps) * int(args.num_generations) :]
                mean_reward = sum(recent) / len(recent) if recent else 0.0
                print(
                    f"step={global_step} epoch={epoch} "
                    f"mean_recent_reward={mean_reward:.4f} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"kl_penalty={float(kl_penalty.detach().cpu()):.6f}"
                )
            if args.save_steps and global_step % int(args.save_steps) == 0:
                _save_checkpoint(
                    checkpoint_dir=output_dir / "checkpoints" / f"step-{global_step}",
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    state={
                        "global_step": global_step,
                        "next_epoch": next_epoch,
                        "next_record_index": next_record_index,
                        "num_train_epochs": float(args.num_train_epochs),
                        "train_file": train_file,
                        "use_residual_pool": bool(args.use_residual_pool),
                        "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
                        "model_name_or_path": args.model_name_or_path,
                        "reference_model_path": reference_model_path,
                        "kl_coef": float(args.kl_coef),
                        "reward_weights": reward_weights,
                        "tau_match": float(args.tau_match),
                        "tau_good": float(args.tau_good),
                        "series_length": int(args.series_length),
                        "max_index": int(args.max_index),
                        "prediction_schema": str(args.prediction_schema),
                    },
                    save_optimizer_state=bool(args.save_optimizer_state),
                )

    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(model_dir)

    eval_summary = None
    if args.eval_file:
        eval_records = load_jsonl(args.eval_file)
        if args.eval_max_samples is not None:
            eval_records = eval_records[: int(args.eval_max_samples)]
        eval_output = Path(args.eval_output) if args.eval_output else output_dir / "eval" / "rl_predictions.jsonl"
        eval_device = _model_device(model)
        eval_summary = _generate_predictions_for_records(
            processor=processor,
            model=model,
            records=eval_records,
            output_path=eval_output,
            max_new_tokens=int(args.eval_max_new_tokens),
            device=eval_device,
        )
        eval_metrics_path = eval_output.with_suffix(".metrics.json")
        eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        eval_metrics_path.write_text(
            json.dumps(eval_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "mode": "grounder_rl_grpo",
        "model_name_or_path": args.model_name_or_path,
        "reference_model_path": reference_model_path if reference_model is not None else None,
        "train_file": train_file,
        "image_root": args.image_root,
        "output_dir": str(output_dir),
        "num_generations": int(args.num_generations),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": float(args.num_train_epochs),
        "kl_coef": float(args.kl_coef),
        "config": args.config,
        "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
        "reward_weights": reward_weights,
        "tau_match": float(args.tau_match),
        "tau_good": float(args.tau_good),
        "series_length": int(args.series_length),
        "max_index": int(args.max_index),
        "prediction_schema": str(args.prediction_schema),
        "use_residual_pool": bool(args.use_residual_pool),
        "residual_pool_file": args.residual_pool_file,
        "residual_pool_epoch_size": residual_pool_epoch_size if args.use_residual_pool else None,
        "residual_pool_sampling_ratios": residual_sampling_ratios if args.use_residual_pool else None,
        "optimizer": str(args.optimizer),
        "gradient_checkpointing": gradient_checkpointing,
        "kl_reference_active": reference_model is not None,
        "kl_estimator": "0.5 * mean((logp_policy - logp_reference)^2)",
        "save_steps": int(args.save_steps),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "optimizer_resumed": optimizer_resumed,
        "save_optimizer_state": bool(args.save_optimizer_state),
        "num_steps": global_step,
        "mean_reward": float(sum(running_rewards) / len(running_rewards)) if running_rewards else 0.0,
        "rollouts_path": str(rollout_path),
        "model_dir": str(model_dir),
        "eval": eval_summary,
    }
    (output_dir / "rl_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
