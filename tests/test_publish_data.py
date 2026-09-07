import importlib.util
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "publish_data", Path(__file__).parents[1] / "scripts/publish_teacher_data.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_removes_credentials_consistently_and_local_provenance():
    secret = "sk-" + "Z" * 24
    data = {
        "result_path": "/home/user/private",
        "session_path": "/tmp/session",
        "a": secret,
        "b": secret,
        "normal": "use tool evidence",
    }
    cleaned = module.scrub(data, Counter())
    assert "result_path" not in cleaned and "session_path" not in cleaned
    assert cleaned["a"] == cleaned["b"]
    assert secret not in str(cleaned)
    assert cleaned["normal"] == data["normal"]


def test_redacts_embedded_secret_assignments_and_direct_fields():
    cleaned = module.scrub(
        {"password": "example-secret", "content": 'password = "example-secret"'}, Counter()
    )
    assert "example-secret" not in str(cleaned)
    assert cleaned["password"] in cleaned["content"]
