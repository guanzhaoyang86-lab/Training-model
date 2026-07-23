from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_exam_data() -> Path:
    return (
        _repo_root().parent
        / "TimeSeriesExam-main"
        / "output"
        / "round_3_folder"
        / "qa_dataset.json"
    )


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base or Path.cwd()) / candidate


def _load_exam_data(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of exam samples in {path}")
    return data


def _stringify_options(options: list[Any]) -> list[str]:
    return [str(option) for option in options]


def _format_options(options: list[Any]) -> str:
    return "\n".join(
        f"{chr(65 + idx)}) {option}" for idx, option in enumerate(_stringify_options(options))
    )


def _format_allowed_final_answers(options: list[Any]) -> str:
    return "\n".join(
        f"Final answer: {chr(65 + idx)}) {option}"
        for idx, option in enumerate(_stringify_options(options))
    )


def _format_indexed_values(name: str, series: list[float], *, precision: int, compact: bool) -> str:
    lines = [f"{name} length: {len(series)}.", f"{name} indexed values:"]
    if compact:
        lines.append(
            ", ".join(f"{idx}:{float(value):.{precision}f}" for idx, value in enumerate(series))
        )
    else:
        lines.extend(f"{idx}: {float(value):.{precision}f}" for idx, value in enumerate(series))
    return "\n".join(lines)


def _render_series(series: list[float], output_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=96)
    ax.plot(range(len(series)), series, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _render_sample_images(sample: dict[str, Any], image_dir: Path, index: int) -> list[Path]:
    if "ts" in sample:
        path = image_dir / f"sample_{index:06d}_ts.png"
        _render_series(sample["ts"], path, title="Time Series")
        return [path]

    paths = [
        image_dir / f"sample_{index:06d}_ts1.png",
        image_dir / f"sample_{index:06d}_ts2.png",
    ]
    _render_series(sample["ts1"], paths[0], title="Time Series 1")
    _render_series(sample["ts2"], paths[1], title="Time Series 2")
    return paths


def _build_prompt(
    sample: dict[str, Any],
    *,
    include_indexed_series_text: bool,
    indexed_series_precision: int,
    indexed_series_compact: bool,
    add_question_hint: bool,
    add_concepts: bool,
) -> str:
    options = _stringify_options(list(sample.get("options", [])))
    options_text = _format_options(options)
    final_answer_choices = _format_allowed_final_answers(options)
    question = str(sample.get("question", "")).strip()
    modality = "one plain time-series plot image" if "ts" in sample else "two plain time-series plot images"

    parts = [
        "Output contract: choose exactly one option and output exactly one line.",
        "Do not include reasoning, calculations, markdown, JSON, or extra text.",
        "",
        f"You are given {modality}. The plot image is plain; use the accompanying indexed numeric values when exact positions matter.",
        "Answer the multiple-choice time-series question.",
        "",
        f"Question:\n{question}",
        "",
        f"Choose from the following options:\n{options_text}",
    ]

    if include_indexed_series_text:
        if "ts" in sample:
            parts.append(
                _format_indexed_values(
                    "Time series",
                    sample["ts"],
                    precision=indexed_series_precision,
                    compact=indexed_series_compact,
                )
            )
        else:
            parts.append(
                _format_indexed_values(
                    "Time series 1",
                    sample["ts1"],
                    precision=indexed_series_precision,
                    compact=indexed_series_compact,
                )
            )
            parts.append(
                _format_indexed_values(
                    "Time series 2",
                    sample["ts2"],
                    precision=indexed_series_precision,
                    compact=indexed_series_compact,
                )
            )

    if add_concepts and sample.get("relevant_concepts"):
        concepts = ", ".join(str(item) for item in sample["relevant_concepts"])
        parts.append(f"Relevant concepts: {concepts}")

    if add_question_hint and sample.get("question_hint"):
        parts.append(f"Hint: {sample['question_hint']}")

    parts.extend(
        [
            "",
            "Your entire response must be exactly one of these lines:",
            final_answer_choices,
        ]
    )
    return "\n".join(parts)


def _build_messages(prompt: str, *, num_images: int) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for _ in range(num_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a careful time-series exam solver. "
                        "Use the image and indexed values to choose the best option. "
                        "Your entire response must be one line in the format "
                        "`Final answer: A) option text`. "
                        "Do not output JSON, markdown, or explanations."
                    ),
                }
            ],
        },
        {"role": "user", "content": user_content},
    ]


def _append_retry_instruction(prompt: str, response: str, options: list[Any]) -> str:
    response_preview = response.strip().replace("\n", " ")[:1000]
    return "\n".join(
        [
            prompt,
            "",
            "The previous response did not follow the required output contract.",
            f"Previous response: {response_preview}",
            "",
            "Fix the format now. Your entire response must be exactly one of these lines:",
            _format_allowed_final_answers(options),
        ]
    )


def _clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    for marker in ("ASSISTANT:", "Assistant:", "<|assistant|>", "<|im_start|>assistant"):
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[-1].strip()
    return cleaned


def _decode_new_tokens(processor, model_inputs: dict[str, Any], generated_ids) -> str:
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


def _generate_response(
    *,
    processor,
    model,
    device,
    images: list[Any],
    messages: list[dict[str, Any]],
    processor_kwargs: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> str:
    import torch

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
        **processor_kwargs,
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generation_kwargs)
    return _decode_new_tokens(processor, model_inputs, outputs)


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


def _model_device(model, *, device_map: str | None):
    import torch

    if device_map:
        return next(
            (param.device for param in model.parameters() if param.device.type != "meta"),
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_answer_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _answer_letter(answer: Any, options: list[Any]) -> str:
    answer_text = _normalize_answer_text(str(answer))
    for idx, option in enumerate(_stringify_options(options)):
        if _normalize_answer_text(option) == answer_text:
            return chr(65 + idx)
    raise ValueError(f"Answer {answer!r} is not present in options: {options!r}")


def _parse_choice_from_line(line: str, options: list[Any]) -> str | None:
    options = _stringify_options(options)
    letters = "".join(chr(65 + idx) for idx in range(len(options)))
    normalized_line = _normalize_answer_text(line)

    for idx, option in enumerate(options):
        letter = chr(65 + idx)
        normalized_option = _normalize_answer_text(option)
        accepted_forms = (
            f"final answer: {letter.lower()}) {normalized_option}",
            f"final answer: {letter.lower()}. {normalized_option}",
            f"{letter.lower()}) {normalized_option}",
            f"{letter.lower()}. {normalized_option}",
        )
        if normalized_line in accepted_forms:
            return letter

    match = re.search(
        rf"(?:^|\b)final\s+answer\s*:\s*([{letters}])(?:\s*[\).:]|\s*$)",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.fullmatch(rf"\s*([{letters}])\s*[\).:]?\s*", line, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    for idx, option in enumerate(options):
        if normalized_line == _normalize_answer_text(option):
            return chr(65 + idx)
    return None


def _is_strict_final_answer_response(response: str, options: list[Any]) -> bool:
    options = _stringify_options(options)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    line = _normalize_answer_text(lines[0])
    for idx, option in enumerate(options):
        letter = chr(65 + idx).lower()
        normalized_option = _normalize_answer_text(option)
        if line == f"final answer: {letter}) {normalized_option}":
            return True
    return False


def _extract_choice(response: str, options: list[Any]) -> str | None:
    options = _stringify_options(options)
    text = response.strip()
    final_lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in reversed(final_lines):
        if "final answer" not in line.lower():
            continue
        parsed = _parse_choice_from_line(line, options)
        if parsed is not None:
            return parsed

    for line in reversed(final_lines[-8:]):
        parsed = _parse_choice_from_line(line, options)
        if parsed is not None:
            return parsed

    normalized_text = _normalize_answer_text(text)
    for idx, option in enumerate(options):
        letter = chr(65 + idx)
        normalized_option = _normalize_answer_text(option)
        if f"{letter.lower()}) {normalized_option}" in normalized_text:
            return letter
        if f"{letter.lower()}. {normalized_option}" in normalized_text:
            return letter
        if f"final answer: {letter.lower()}" in normalized_text:
            return letter

    for idx, option in enumerate(options):
        if _normalize_answer_text(option) in normalized_text:
            return chr(65 + idx)
    return None


def _evaluate_response(sample: dict[str, Any], response: str) -> tuple[str | None, bool]:
    options = list(sample["options"])
    expected = _answer_letter(sample["answer"], options)
    predicted = _extract_choice(response, options)
    return predicted, predicted == expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL checkpoint on TimeSeriesExam round 3.")
    parser.add_argument("--data-file", type=str, default=str(_default_exam_data()))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/timeseries_exam_round3_predictions.json")
    parser.add_argument("--image-cache-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=131072)
    parser.add_argument("--indexed-series-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--indexed-series-precision", type=int, default=4)
    parser.add_argument("--indexed-series-compact", action="store_true")
    parser.add_argument("--add-question-hint", action="store_true")
    parser.add_argument("--add-concepts", action="store_true")
    parser.add_argument("--retry-bad-format", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo_root = _repo_root()
    data_file = _resolve_path(args.data_file, base=repo_root)
    model_path = _resolve_path(args.model_path, base=repo_root)
    output_path = _resolve_path(args.output, base=repo_root)
    image_cache_dir = (
        _resolve_path(args.image_cache_dir, base=repo_root)
        if args.image_cache_dir
        else output_path.with_suffix("") / "images"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor_kwargs = {"max_pixels": args.max_pixels}
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        **processor_kwargs,
    )

    model_kwargs: dict[str, Any] = {}
    dtype = _resolve_dtype(args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    device_map = None if args.device_map.lower() in {"", "none"} else args.device_map
    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        **model_kwargs,
    )
    if not device_map:
        device = _model_device(model, device_map=None)
        model = model.to(device)
    else:
        device = _model_device(model, device_map=device_map)
    model.eval()

    samples = _load_exam_data(data_file)
    if args.start_index < 0 or args.start_index >= len(samples):
        raise ValueError(f"--start-index must be in [0, {len(samples) - 1}]")
    selected = samples[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    results = []
    correct = 0
    generation_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature})
    else:
        generation_kwargs["do_sample"] = False

    for offset, sample in enumerate(selected, start=args.start_index):
        image_paths = _render_sample_images(sample, image_cache_dir, offset + 1)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompt = _build_prompt(
            sample,
            include_indexed_series_text=args.indexed_series_text,
            indexed_series_precision=args.indexed_series_precision,
            indexed_series_compact=args.indexed_series_compact,
            add_question_hint=args.add_question_hint,
            add_concepts=args.add_concepts,
        )
        messages = _build_messages(prompt, num_images=len(images))
        response = _generate_response(
            processor=processor,
            model=model,
            device=device,
            images=images,
            messages=messages,
            processor_kwargs=processor_kwargs,
            generation_kwargs=generation_kwargs,
        )
        predicted, is_correct = _evaluate_response(sample, response)
        raw_response = response
        retried = False
        strict_format = _is_strict_final_answer_response(response, list(sample["options"]))

        if args.retry_bad_format and not strict_format:
            retry_generation_kwargs = dict(generation_kwargs)
            retry_generation_kwargs["max_new_tokens"] = min(int(args.max_new_tokens), 64)
            retry_prompt = _append_retry_instruction(prompt, response, list(sample["options"]))
            retry_messages = _build_messages(retry_prompt, num_images=len(images))
            retry_response = _generate_response(
                processor=processor,
                model=model,
                device=device,
                images=images,
                messages=retry_messages,
                processor_kwargs=processor_kwargs,
                generation_kwargs=retry_generation_kwargs,
            )
            retry_predicted, retry_is_correct = _evaluate_response(sample, retry_response)
            if retry_predicted is not None:
                response = retry_response
                predicted = retry_predicted
                is_correct = retry_is_correct
                strict_format = _is_strict_final_answer_response(response, list(sample["options"]))
            retried = True

        correct += int(is_correct)

        result = {
            "index": offset,
            "id": sample.get("id"),
            "category": sample.get("category"),
            "subcategory": sample.get("subcategory"),
            "difficulty": sample.get("difficulty"),
            "question_type": sample.get("question_type"),
            "question": sample.get("question"),
            "options": sample.get("options"),
            "answer": sample.get("answer"),
            "expected_letter": _answer_letter(sample["answer"], list(sample["options"])),
            "predicted_letter": predicted,
            "correct": is_correct,
            "response": response,
            "raw_response": raw_response,
            "retried": retried,
            "strict_format": strict_format,
            "image_paths": [str(path) for path in image_paths],
        }
        results.append(result)

        running_total = len(results)
        print(
            f"[{running_total}/{len(selected)}] "
            f"id={sample.get('id')} predicted={predicted} correct={is_correct} "
            f"running_accuracy={correct / running_total:.4f}",
            flush=True,
        )

    summary = {
        "accuracy": correct / len(results) if results else 0.0,
        "correct": correct,
        "total": len(results),
        "data_file": str(data_file),
        "model_path": str(model_path),
        "image_cache_dir": str(image_cache_dir),
        "generation": {
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "max_pixels": args.max_pixels,
        },
        "prompt_options": {
            "indexed_series_text": args.indexed_series_text,
            "indexed_series_precision": args.indexed_series_precision,
            "indexed_series_compact": args.indexed_series_compact,
            "add_question_hint": args.add_question_hint,
            "add_concepts": args.add_concepts,
            "retry_bad_format": args.retry_bad_format,
        },
    }

    payload = {"summary": summary, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(output_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {summary['accuracy']:.4f} ({correct}/{len(results)})")
    print(f"Predictions: {output_path}")
    print(f"Metrics: {output_path.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
