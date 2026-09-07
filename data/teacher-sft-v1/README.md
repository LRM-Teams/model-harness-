# ClawEval DeepSeek V4 Flash teacher data — adapted SFT v1

Derived from the user's existing DeepSeek V4 Flash ClawEval runs and published at the user's request. This dataset is an adapted export, not an exact capture of historical model requests.

- 228 structurally accepted episodes; 1,034 training samples and 91 validation samples.
- Complete history as prompt, one assistant turn as completion, cached tool JSON schemas.
- Thinking blocks removed. A versioned generic system prompt replaces unreliable historical prompt remnants.
- Splits grouped by task suffix family, including known bilingual counterparts.
- Credential-like strings and explicit secret assignments replaced with stable placeholders; host provenance paths removed. Repeated references remain consistent. Pattern screening is not a comprehensive privacy audit.
- Model tokenizer: Qwen/Qwen3.5-0.8B, revision in `report.json`; sample lengths recomputed after sanitization.
- Score/grader feedback is not in model inputs. Passing grades are a selection signal, not proof that every action is correct.
- Source/provenance hashes and exclusion reasons are retained for reproducibility. Raw teacher artifacts and local authentication files are not included.

Files: `train.jsonl`, `validation.jsonl`, `episodes.jsonl`, `rejected.jsonl`, `overlength.jsonl`, `report.json`. Check file hashes against the report before training. The report's original `counts` and length distribution refer to the pre-publication export; `publication.sanitization` and each row's `num_tokens` describe this sanitized release.

This export does not grant additional rights to underlying ClawEval tasks, retrieved content, or third-party materials; source terms continue to apply. Do not use tasks seen in training as independent benchmark tests.

See [training instructions](../../docs/sft-running.md) and [data preparation details](../../docs/teacher-sft.md).
