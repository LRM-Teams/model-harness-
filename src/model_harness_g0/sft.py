"""Single-GPU text SFT with explicit completion masks and before/after checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from .storage import write_json


def encode_sample(tokenizer, row, max_length):
    options = dict(tools=row["tools"], tokenize=True, return_dict=False, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, **options)
    ids = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"], add_generation_prompt=False, **options
    )
    if not isinstance(ids, list) or not isinstance(prompt, list):
        raise ValueError("Tokenizer must return token ID lists")
    if ids[: len(prompt)] != prompt:
        raise ValueError(f"Non-prefix-preserving chat template: {row['sample_id']}")
    if len(ids) <= len(prompt) or len(row["completion"]) != 1:
        raise ValueError("Expected exactly one nonempty assistant completion")
    if row["completion"][0]["role"] != "assistant":
        raise ValueError("Only assistant outputs may be supervised")
    if len(ids) > max_length:
        return None
    labels = [-100] * len(prompt) + ids[len(prompt) :]
    assert all(x == -100 for x in labels[: len(prompt)])
    return {
        "input_ids": ids,
        "labels": labels,
        "prompt_ids": prompt,
        "sample_id": row["sample_id"],
        "task_id": row["task_id"],
        "family": row["family"],
        "target_tokens": len(ids) - len(prompt),
        "expected_tool_names": [
            x["function"]["name"] for x in row["completion"][0].get("tool_calls", [])
        ],
    }


def load_data(path, tokenizer, max_length):
    report = json.loads((path / "report.json").read_text())
    for name, expected in report["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    splits, stats, families = {}, {}, {}
    for split in ["train", "validation"]:
        rows = [json.loads(x) for x in (path / f"{split}.jsonl").read_text().splitlines()]
        families[split] = {r["family"] for r in rows}
        encoded = [encode_sample(tokenizer, r, max_length) for r in rows]
        splits[split] = [r for r in encoded if r is not None]
        stats[split] = {
            "source_samples": len(rows),
            "used_samples": len(splits[split]),
            "over_length": sum(r is None for r in encoded),
            "target_tokens": sum(r["target_tokens"] for r in splits[split]),
        }
        if not splits[split]:
            raise ValueError(f"No usable {split} samples")
    if families["train"] & families["validation"]:
        raise ValueError("Task family leakage")
    return splits, stats


def stratified(rows, limit):
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["sample_id"]):
        groups[r["task_id"]].append(r)
    result = []
    while groups and (not limit or len(result) < limit):
        for task in sorted(list(groups)):
            result.append(groups[task].pop(0))
            if not groups[task]:
                del groups[task]
            if limit and len(result) >= limit:
                break
    return result


def batch(row, device):
    import torch

    return {
        k: torch.tensor([row[k]], dtype=torch.long, device=device) for k in ["input_ids", "labels"]
    } | {"attention_mask": torch.ones((1, len(row["input_ids"])), dtype=torch.long, device=device)}


def evaluate(model, rows):
    import torch

    model.eval()
    by_task = defaultdict(lambda: [0.0, 0])
    entries = []
    with torch.no_grad():
        for row in rows:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch(row, "cuda"), use_cache=False).loss.float().item()
            if not math.isfinite(loss):
                raise ValueError("Nonfinite evaluation loss")
            n = row["target_tokens"]
            by_task[row["task_id"]][0] += loss * n
            by_task[row["task_id"]][1] += n
            entries.append({"sample_id": row["sample_id"], "nll": loss, "tokens": n})
    return {
        "token_mean_nll": sum(v[0] for v in by_task.values()) / sum(v[1] for v in by_task.values()),
        "task_mean_nll": sum(v[0] / v[1] for v in by_task.values()) / len(by_task),
        "samples": len(rows),
        "tasks": len(by_task),
        "entries": entries,
    }


def generate_probes(model, tokenizer, rows, limit):
    import torch

    model.eval()
    result = []
    for row in stratified(rows, limit) if limit else []:
        ids = torch.tensor([row["prompt_ids"]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = out[0, ids.shape[1] :].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=False)
        names = re.findall(r"<function=([^>\n]+)>", text)
        result.append(
            {
                "sample_id": row["sample_id"],
                "text": text,
                "expected_tool_names": row["expected_tool_names"],
                "predicted_tool_names": names,
                "tool_name_sequence_match": names == row["expected_tool_names"],
                "hit_generation_limit": len(generated) == 256,
                "note": "Offline teacher-prefix probe; not a live task success score.",
            }
        )
    return result


def train(args, tokenizer, splits, output):
    import gc

    import torch
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a BF16 CUDA GPU; data checks do not")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    eval_rows = stratified(splits["validation"], 8 if args.mode == "smoke" else 0)
    train_rows = stratified(splits["train"], 32 if args.mode == "smoke" else 0)
    write_json(
        output / "selected_samples.json",
        {
            "train": [x["sample_id"] for x in train_rows],
            "eval": [x["sample_id"] for x in eval_rows],
        },
    )
    # FP32 master parameters and optimizer states, BF16 autocast computation.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    baseline = evaluate(model, eval_rows)
    probes_before = generate_probes(model, tokenizer, eval_rows, args.probes)
    write_json(output / "baseline.json", baseline)
    write_json(output / "probes_before.json", probes_before)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    groups = defaultdict(list)
    for row in train_rows:
        groups[row["task_id"]].append(row)
    group_names = sorted(groups)
    # Track multiple text tensors that actually received gradients.
    initial = {
        name: p.detach().flatten()[:128].cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    losses = []
    optimizer.zero_grad(set_to_none=True)
    with (output / "train_log.jsonl").open("x") as log:
        for step in range(args.steps):
            model.train()
            values, sample_ids = [], []
            for _ in range(args.grad_accum):
                row = random.choice(groups[random.choice(group_names)])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch(row, "cuda"), use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                values.append(loss.detach().float().item())
                sample_ids.append(row["sample_id"])
                (loss / args.grad_accum).backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
            if norm == 0:
                raise ValueError("No gradient reached trainable parameters")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            entry = {
                "optimizer_step": step + 1,
                "loss": sum(values) / len(values),
                "gradient_norm": norm,
                "sample_ids": sample_ids,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            losses.append(entry["loss"])
            if step == 0 or (step + 1) % 5 == 0:
                print(json.dumps(entry | {"sample_ids": "see train_log.jsonl"}), flush=True)
    changed = any(
        not torch.equal(initial[name], p.detach().flatten()[:128].cpu())
        for name, p in model.named_parameters()
        if name in initial
    )
    if not changed:
        raise ValueError("No sampled trainable weights changed")
    after = evaluate(model, eval_rows)
    probes_after = generate_probes(model, tokenizer, eval_rows, args.probes)
    write_json(output / "after.json", after)
    write_json(output / "probes_after.json", probes_after)
    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    reference = evaluate(model, eval_rows[:1])["token_mean_nll"]
    peak = torch.cuda.max_memory_allocated()
    # Release references held by optimizer, parameter list, and last loss graph.
    del optimizer, params, model, loss, param
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
    reloaded_loss = evaluate(reloaded, eval_rows[:1])["token_mean_nll"]
    delta = abs(reference - reloaded_loss)
    result = {
        "pipeline_passed": changed and delta <= 1e-4,
        "optimizer_steps": args.steps,
        "weights_changed": changed,
        "reload_nll_abs_delta": delta,
        "peak_allocated_gb": peak / 1024**3,
        "validation_nll_before": baseline["token_mean_nll"],
        "validation_nll_after": after["token_mean_nll"],
        "task_mean_nll_before": baseline["task_mean_nll"],
        "task_mean_nll_after": after["task_mean_nll"],
        "validation_nll_decreased": after["token_mean_nll"] < baseline["token_mean_nll"],
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "task_effectiveness_proven": False,
        "reason": "Requires paired live held-out task evaluation with the same harness.",
        "checkpoint": str(checkpoint),
        "mode": args.mode,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output / "training_report.json", result)
    print(json.dumps(result, indent=2))
    if not result["pipeline_passed"]:
        raise RuntimeError("Checkpoint reload check failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/teacher-sft-v1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--check-data-only", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--revision")
    p.add_argument("--steps", type=int)
    p.add_argument("--max-length", type=int)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", type=int, default=4)
    args = p.parse_args()
    args.steps = args.steps if args.steps is not None else (20 if args.mode == "smoke" else 200)
    args.max_length = args.max_length or (2048 if args.mode == "smoke" else 8192)
    if min(args.steps, args.grad_accum, args.max_length) <= 0 or args.probes < 0:
        raise ValueError("Steps, grad accumulation and length must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    report = json.loads((args.data / "report.json").read_text())
    args.revision = args.revision or report["tokenizer"]["revision"]
    if (
        args.model != report["tokenizer"]["repo"]
        or args.revision != report["tokenizer"]["revision"]
    ):
        raise ValueError("Model and tokenizer must match the dataset lock")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    splits, stats = load_data(args.data, tokenizer, args.max_length)
    args.output.mkdir(parents=True)
    write_json(
        args.output / "data_check.json",
        {
            "passed": True,
            "splits": stats,
            "max_length": args.max_length,
            "loss": "only final assistant completion",
            "prefix_ids_verified": True,
            "model_revision": args.revision,
            "dataset_report_sha256": hashlib.sha256(
                (args.data / "report.json").read_bytes()
            ).hexdigest(),
        },
    )
    write_json(
        args.output / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    if args.check_data_only:
        print(json.dumps(stats, indent=2))
        return
    write_json(
        args.output / "training_report.json", {"pipeline_passed": False, "status": "RUNNING"}
    )
    train(args, tokenizer, splits, args.output)


if __name__ == "__main__":
    main()
