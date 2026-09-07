from model_harness_g0.parity import verify
from model_harness_g0.storage import digest, write_json


def test_mock_and_hf_only_cannot_pass_gate(tmp_path):
    report = {
        "backend": "mock",
        "collection_checks_passed": True,
        "optimizer_steps": 0,
        "parser_check": {"passed": True},
        "model_revision": "rev",
    }
    write_json(tmp_path / "compatibility_report.json", report)
    assert not verify(tmp_path)["g0_passed"]
    report["backend"] = "areal-sglang"
    write_json(tmp_path / "compatibility_report.json", report)
    write_json(
        tmp_path / "parity_report.json",
        {
            "passed": True,
            "reference": "hf",
            "model_revision": "rev",
            "collection_report_hash": digest(report),
            "trace_hashes": {str(i): "hash" for i in range(6)},
        },
    )
    assert not verify(tmp_path)["g0_passed"]
