"""RUN_PI_E2E=1 enables a real Pi process test with a synthetic model backend."""

import asyncio
import os
import shutil
import socket
import sys

import httpx
import pytest

from model_harness_g0.parity import verify
from model_harness_g0.runner import smoke


@pytest.mark.skipif(os.environ.get("RUN_PI_E2E") != "1", reason="opt-in real Pi test")
async def test_real_pi_20_sequential_4_concurrent(tmp_path):
    pi = os.environ.get("PI_BIN") or shutil.which("pi")
    assert pi, "install Pi 0.84.3 or set PI_BIN"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    key = "g0-e2e-fixture-key-not-a-real-secret"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "model_harness_g0.cli",
        "serve",
        "--backend",
        "mock",
        "--state",
        str(tmp_path / "server"),
        "--port",
        str(port),
        env={**os.environ, "G0_ADMIN_KEY": key},
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with httpx.AsyncClient() as c:
            for _ in range(100):
                if proc.returncode is not None:
                    pytest.fail("fixture server exited")
                try:
                    if (await c.get(url + "/health")).is_success:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.1)
            else:
                pytest.fail("fixture server did not start")
        report = await smoke(url, key, tmp_path / "run", pi)
        assert report["collection_checks_passed"], report["results"]
        assert report["user_config_unchanged"]
        assert report["active_sessions_after"] == 0
        assert len(report["results"]) == 24
        assert all(r["task_success"] for r in report["results"])
        assert not list((tmp_path / "run").glob("*/agent"))
        assert not verify(tmp_path / "run")["g0_passed"]
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except TimeoutError:
            proc.kill()
            await proc.wait()
