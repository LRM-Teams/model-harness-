from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .prepare import check_lock, doctor
from .storage import digest, write_json
from .trace import validate_trace


class Reference:
    def __init__(self, model_path, device, kind):
        import torch

        self.kind, self.device = kind, device
        self.engine = None
        self.group_dir = None
        if kind == "fsdp":
            import torch.distributed as dist
            from areal.api import FinetuneSpec
            from areal.engine.fsdp_engine import FSDPEngine

            if dist.is_initialized():
                raise ValueError("Run parity as a dedicated process, not inside a trainer")
            self.group_dir = tempfile.TemporaryDirectory(prefix="g0-fsdp-")
            os.environ["LOCAL_RANK"] = str(torch.device(device).index or 0)
            torch.cuda.set_device(device)
            dist.init_process_group(
                "nccl",
                init_method="file://" + self.group_dir.name + "/rendezvous",
                rank=0,
                world_size=1,
            )
            try:
                self.engine = FSDPEngine.from_pretrained(
                    model_path,
                    experiment_name="g0",
                    trial_name="parity",
                    learning_rate=None,
                    dtype="bfloat16",
                    disable_dropout=True,
                    attn_impl="eager",
                )
                self.engine.create_process_group()
                self.engine.initialize(
                    addr=None,
                    ft_spec=FinetuneSpec(total_train_epochs=0, dataset_size=1, train_batch_size=1),
                )
                if getattr(self.engine, "optimizer", None) is not None:
                    raise RuntimeError("G0 must not create an optimizer")
                self.engine.eval()
                self.model = self.engine.model
            except BaseException:
                if dist.is_initialized():
                    dist.destroy_process_group()
                self.group_dir.cleanup()
                raise
        else:
            from transformers import AutoModelForImageTextToText

            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                    local_files_only=True,
                )
                .to(device)
                .eval()
            )

    def logprobs(self, inp, out):
        import torch

        ids = torch.tensor([inp + out], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self.engine is not None:
                logp = self.engine.forward_batch(
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
                )
                if not torch.is_tensor(logp) or logp.numel() != ids.numel():
                    raise ValueError("Unexpected AReaL forward logprob shape")
                return logp.reshape(-1)[len(inp) - 1 : len(inp) + len(out) - 1].float()
            result = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=len(out) + 1,
            )
            logits = result.logits[0, -len(out) - 1 : -1].float()
            targets = torch.tensor(out, device=self.device)
            return logits.gather(1, targets[:, None]).squeeze(1) - logits.logsumexp(-1)

    def close(self):
        if self.engine is not None:
            import torch.distributed as dist

            self.engine.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()
            self.group_dir.cleanup()


def parity(run: Path, lock_path: Path, device="cuda:0", reference="fsdp"):
    import torch

    report = json.loads((run / "compatibility_report.json").read_text())
    lock = json.loads(lock_path.read_text())
    if report["backend"] != "areal-sglang" or report["model_revision"] != lock["model_revision"]:
        raise ValueError("parity requires real AReaL traces matching the locked model")
    check_lock(lock)
    chosen = {}
    for r in report["results"]:
        if r["protocol_ok"] and r["task_success"]:
            chosen.setdefault(r["case"], r["trial_id"])
    if len(chosen) != 6:
        raise ValueError("need successful traces for all six cases before parity")
    output = run / ("parity_report.json" if reference == "fsdp" else "hf_parity_report.json")
    # A failed repeat must not leave behind an old passing report.
    write_json(output, {"passed": False, "status": "RUNNING", "reference": reference})
    ref = Reference(lock["snapshot_path"], device, reference)
    entries, trace_hashes = [], {}
    try:
        inventory = {
            "reference": reference,
            "linear_modules": [
                {
                    "name": name,
                    "shape": list(module.weight.shape),
                    "eligible_for_later_lora_scan": not any(
                        s in name.lower() for s in ("visual", "vision", "embed", "lm_head")
                    ),
                }
                for name, module in ref.model.named_modules()
                if isinstance(module, torch.nn.Linear)
            ],
            "note": "Inventory only; G0 does not load or train LoRA adapters.",
        }
        write_json(run / "module_inventory.json", inventory)
        for case, trial in chosen.items():
            episode = json.loads((run / trial / "episode.json").read_text())
            trace_hashes[trial] = digest(episode)
            for row in episode["interactions"]:
                validate_trace(row)
                if row["synthetic"] or row["model_revision"] != lock["model_revision"]:
                    raise ValueError("invalid parity input")
                new = ref.logprobs(row["input_ids"], row["output_ids"])
                old = torch.tensor(row["old_logprobs"], device=new.device)
                diff = (new - old).abs()
                if not torch.isfinite(diff).all():
                    raise ValueError("nonfinite logprob difference")
                mean, p99 = diff.mean().item(), diff.quantile(0.99).item()
                entries.append(
                    {
                        "case": case,
                        "completion_id": row["completion_id"],
                        "mean_abs_delta": mean,
                        "p99_abs_delta": p99,
                        "tokens": len(row["output_ids"]),
                        "passed": mean <= 0.02 and p99 <= 0.1,
                    }
                )
    finally:
        ref.close()
    result = {
        "schema_version": 1,
        "reference": reference,
        "optimizer_steps": 0,
        "model_revision": lock["model_revision"],
        "collection_report_hash": digest(report),
        "trace_hashes": trace_hashes,
        "mean_tolerance": 0.02,
        "p99_tolerance": 0.1,
        "passed": bool(entries) and all(e["passed"] for e in entries),
        "entries": entries,
        "environment": doctor(),
    }
    write_json(output, result)
    return result


def verify(run):
    report = json.loads((run / "compatibility_report.json").read_text())
    path = run / "parity_report.json"
    checked = json.loads(path.read_text()) if path.exists() else {}
    passed = (
        report["backend"] == "areal-sglang"
        and report["collection_checks_passed"]
        and report.get("parser_check", {}).get("passed", False)
        and report["optimizer_steps"] == 0
        and checked.get("passed", False)
        and checked.get("reference") == "fsdp"
        and checked.get("collection_report_hash") == digest(report)
        and checked.get("model_revision") == report["model_revision"]
        and len(checked.get("trace_hashes", {})) == 6
    )
    if passed:
        for trial, expected in checked["trace_hashes"].items():
            if digest(json.loads((run / trial / "episode.json").read_text())) != expected:
                passed = False
    result = {
        "g0_passed": bool(passed),
        "collection": report["collection_checks_passed"],
        "parity": checked.get("passed", False),
        "optimizer_steps": 0,
        "scope": "synthetic tasks, Pi RPC, AReaL collection + FSDP forward parity",
        "not_validated": ["SFT/GRPO updates", "benchmark GT boundary", "LoRA reload"],
    }
    write_json(run / "gate_report.json", result)
    return result
