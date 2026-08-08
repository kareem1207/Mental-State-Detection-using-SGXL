"""Launches and manages the local llama-server.exe (Qwen GGUF) subprocess."""
from __future__ import annotations

import subprocess
import time

import httpx

from config import settings

_process: subprocess.Popen | None = None


def is_server_up(timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(f"{settings.llama_server_base_url}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_llama_server_running(ready_timeout: float = 120.0) -> subprocess.Popen | None:
    """Starts llama-server.exe if it isn't already serving. Returns the Popen handle
    if this call started it, or None if a server was already up."""
    global _process

    if is_server_up():
        return None

    if not settings.llama_server_exe.exists():
        raise FileNotFoundError(f"llama-server.exe not found at {settings.llama_server_exe}")
    if not settings.qwen_model_path.exists():
        raise FileNotFoundError(f"Model not found at {settings.qwen_model_path}")

    _process = subprocess.Popen(
        [
            str(settings.llama_server_exe),
            "-m", str(settings.qwen_model_path),
            "--host", settings.llama_server_host,
            "--port", str(settings.llama_server_port),
            "--jinja",
        ],
        cwd=str(settings.llama_bin_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if is_server_up():
            return _process
        if _process.poll() is not None:
            output = _process.stdout.read() if _process.stdout else ""
            raise RuntimeError(f"llama-server.exe exited early (code {_process.returncode}):\n{output}")
        time.sleep(0.5)

    stop()
    raise TimeoutError(f"llama-server.exe did not become ready within {ready_timeout}s")


def stop() -> None:
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None


if __name__ == "__main__":
    proc = ensure_llama_server_running()
    if proc is None:
        print(f"llama-server already running at {settings.llama_server_base_url}")
    else:
        print(f"llama-server started (pid={proc.pid}) at {settings.llama_server_base_url}")
        print("Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            stop()
