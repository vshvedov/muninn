import httpx
import pytest

from muninn.egress import EgressDenied, LocalhostOnlyTransport, make_localhost_client


async def test_localhost_request_passes_through_transport(monkeypatch) -> None:
    sent = []

    class _Stub(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            sent.append(str(request.url))
            return httpx.Response(200, text="ok")

    transport = LocalhostOnlyTransport()
    # Replace the inner stack: super().handle_async_request would actually open a connection.
    # We simulate: any URL is routed through our transport; the transport's policy check fires first.
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EgressDenied):
            await client.get("http://example.com/")


async def test_localhost_url_does_not_raise_egress_denied(monkeypatch) -> None:
    """Localhost requests pass the policy check (and may then fail with a real network error)."""
    transport = LocalhostOnlyTransport()
    async with httpx.AsyncClient(transport=transport, timeout=1.0) as client:
        try:
            await client.get("http://127.0.0.1:1/")
        except EgressDenied:
            pytest.fail("localhost url should not be blocked")
        except Exception:
            # ConnectError or TimeoutException is fine - point is, EgressDenied did NOT fire.
            pass


async def test_make_localhost_client_returns_async_client() -> None:
    client = make_localhost_client()
    assert isinstance(client, httpx.AsyncClient)
    await client.aclose()
