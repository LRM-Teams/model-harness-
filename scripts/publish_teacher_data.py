"""Package an explicitly authorized dataset for Git; remove credential-like strings and host paths."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PATTERNS = [
    r"\bsk-[A-Za-z0-9_-]{16,}",
    r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}",
    r"\bAKIA[A-Z0-9]{16}\b",
    r"(?i)bearer\s+[A-Za-z0-9_.-]{20,}",
]


def replacement(value, counts):
    counts["credential_occurrences"] += 1
    return "REDACTED_CREDENTIAL_" + hashlib.sha256(value.encode()).hexdigest()[:12]


def scrub(value, counts):
    if isinstance(value, dict):
        return {
            k: (
                replacement(v, counts)
                if k.lower()
                in {
                    "api_key",
                    "password",
                    "access_token",
                    "client_secret",
                    "secret_access_key",
                    "aws_secret_access_key",
                }
                and isinstance(v, str)
                and v
                else scrub(v, counts)
            )
            for k, v in value.items()
            if k not in {"result_path", "session_path"}
        }
    if isinstance(value, list):
        return [scrub(v, counts) for v in value]
    if not isinstance(value, str):
        return value
    for pattern in PATTERNS:

        def replace(match):
            counts["credential_occurrences"] += 1
            return "REDACTED_CREDENTIAL_" + hashlib.sha256(match[0].encode()).hexdigest()[:12]

        value = re.sub(pattern, replace, value)
    value = re.sub(
        r"(?i)([\"']?(?:aws_secret_access_key|secret_access_key|api_key|access_token|password|client_secret)[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9_./+=-]{8,})",
        lambda m: (
            m[1] + (m[2] if m[2].startswith("REDACTED_CREDENTIAL_") else replacement(m[2], counts))
        ),
        value,
    )
    return value


def main():
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    report = json.loads((args.source / "report.json").read_text())
    info = report["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(info["repo"], revision=info["revision"])
    args.output.mkdir(parents=True)
    counts = Counter()
    for name in [
        "train.jsonl",
        "validation.jsonl",
        "episodes.jsonl",
        "overlength.jsonl",
        "rejected.jsonl",
    ]:
        with (args.output / name).open("x") as out:
            for line in (args.source / name).read_text().splitlines():
                row = scrub(json.loads(line), counts)
                if "prompt" in row:
                    ids = tokenizer.apply_chat_template(
                        row["prompt"] + row["completion"],
                        tools=row["tools"],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    row["num_tokens"] = len(ids)
                    if len(ids) > report["max_length"]:
                        counts["excluded_after_redaction"] += 1
                        continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[name] += 1
    report["publication"] = {
        "sanitization": dict(counts),
        "note": "Credential-like strings replaced consistently; host provenance paths removed. "
        "This is pattern screening, not a guarantee that every sensitive string is detected.",
    }
    report["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report["publication"], indent=2))


if __name__ == "__main__":
    main()
