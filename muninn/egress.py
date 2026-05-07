"""Asserting httpx transport: blocks any request whose host is not localhost.

Phase 1 scope: this transport is wrapped around the single shared httpx.AsyncClient
that we hand to OllamaProvider. We do not enable OTel exporters or any other
network code in Phase 1, so that single client is the only egress point.
"""
from __future__ import annotations

import httpx

ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class EgressDenied(RuntimeError):
    """Raised when an HTTP request targets a non-allowed host."""


class LocalhostOnlyTransport(httpx.AsyncHTTPTransport):
    """An async HTTP transport that refuses to talk to anything but localhost."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").lower()
        if host not in ALLOWED_HOSTS:
            raise EgressDenied(
                f"refusing non-localhost request to {host!r} ({request.url})"
            )
        return await super().handle_async_request(request)


def make_localhost_client() -> httpx.AsyncClient:
    """Construct an httpx.AsyncClient pinned to localhost via LocalhostOnlyTransport.

    Timeout strategy:
    - connect: 10s. Fail fast when Ollama is down / wrong base_url. The
      health-check probe relies on this; we want a tight connect bound so
      the user sees the red banner quickly.
    - read: None. LLM streams can pause for tens of seconds between chunks
      (model thinking after a tool call, GPU contention, low quantization
      slowness). An arbitrary read cap turns those normal pauses into
      `/bug failed: ReadTimeout` style errors. The user has Esc to cancel
      a stream they consider stuck; we don't need a wall-clock cap on top
      of that.
    - write: 60s. Sending a request body should never take long on
      localhost. If it does, something is wrong worth surfacing.
    - pool: 60s. Connection-pool acquisition. Same rationale as write.
    """
    return httpx.AsyncClient(
        transport=LocalhostOnlyTransport(),
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0),
    )
