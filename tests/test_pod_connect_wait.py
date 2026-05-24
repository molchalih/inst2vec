from __future__ import annotations

import httpx
import pytest

from modules.embeddings.pod import _coordinator_base_url, wait_for_coordinator


def test_coordinator_base_url_defaults_to_http_for_bare_host():
    # Raw port-forward: COORDINATOR_PUBLIC_HOST is a bare host:port.
    assert _coordinator_base_url("1.2.3.4:8765") == "http://1.2.3.4:8765"


def test_coordinator_base_url_honors_explicit_scheme():
    # TLS tunnel (Cloudflare/ngrok): the scheme is set, port 443 implied.
    assert (
        _coordinator_base_url("https://abc.trycloudflare.com")
        == "https://abc.trycloudflare.com"
    )
    assert _coordinator_base_url("http://orch:8765") == "http://orch:8765"


def test_returns_once_healthz_is_up():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    wait_for_coordinator("http://orch:8765", timeout_s=5, poll_s=0.0, _client=client)
    assert state["n"] == 3


def test_raises_systemexit_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SystemExit, match="not reachable"):
        wait_for_coordinator(
            "http://orch:8765", timeout_s=0.05, poll_s=0.01, _client=client
        )
