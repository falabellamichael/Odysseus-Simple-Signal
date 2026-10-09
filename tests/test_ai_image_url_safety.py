import base64
import ipaddress
import threading
from unittest.mock import MagicMock

import pytest

from src import ai_interaction


class _GenerationResponse:
    status_code = 200
    text = ""

    def __init__(self, image_url, *, dashscope=False, b64_image=None):
        self._image_url = image_url
        self._dashscope = dashscope
        self._b64_image = b64_image

    def json(self):
        if self._dashscope:
            return {"output": {"choices": [{"message": {"content": [{"image": self._image_url}]}}]}}
        if self._b64_image is not None:
            return {"data": [{"b64_json": self._b64_image}]}
        return {"data": [{"url": self._image_url}]}


class _DownloadResponse:
    status_code = 503
    content = b""


def _patch_generation(monkeypatch, image_url, *, dashscope=False, b64_image=None, posted=None, public_url=""):
    async def _post(self, url, json, headers):
        if posted is not None:
            posted.append((url, json))
        return _GenerationResponse(image_url, dashscope=dashscope, b64_image=b64_image)

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        post = _post

    import httpx
    import src.settings as settings

    monkeypatch.setattr(settings, "load_settings", lambda: {})
    monkeypatch.setattr(
        settings, "get_setting",
        lambda key, default=None: public_url if key == "app_public_url" else default,
    )
    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(
        ai_interaction,
        "_resolve_model",
        lambda model_spec, owner=None, **kwargs: (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            if dashscope else "https://api.openai.example/v1/chat/completions",
            "qwen-image" if dashscope else "dall-e-3",
            {"Authorization": "Bearer test"},
        ),
    )


async def test_generate_image_validates_provider_url_before_download(monkeypatch):
    import httpx
    import src.url_safety as url_safety

    provider_url = "https://images.example.com/generated.png?sig=abc"
    events = []
    _patch_generation(monkeypatch, provider_url)

    def _check_outbound_url(url, *, block_private=False):
        events.append(("check", url, block_private))
        return True, "ok"

    def _get(url, *, timeout):
        events.append(("get", url, timeout))
        return _DownloadResponse()

    monkeypatch.setattr(url_safety, "check_outbound_url", _check_outbound_url)
    monkeypatch.setattr(httpx, "get", _get)

    result = await ai_interaction.do_generate_image("draw a chair\ndall-e-3")

    assert result["image_url"] == provider_url
    assert events == [
        ("check", provider_url, False),
        ("get", provider_url, 60),
    ]


async def test_generate_image_rejects_unsafe_provider_url_without_download(monkeypatch):
    import httpx
    import src.url_safety as url_safety

    unsafe_url = "http://169.254.169.254/latest/meta-data"
    events = []
    _patch_generation(monkeypatch, unsafe_url)

    def _check_outbound_url(url, *, block_private=False):
        events.append(("check", url, block_private))
        return False, "link-local address blocked (SSRF metadata risk): 169.254.169.254"

    def _get(url, *, timeout):
        raise AssertionError("unsafe provider image URL must not be downloaded")

    monkeypatch.setattr(url_safety, "check_outbound_url", _check_outbound_url)
    monkeypatch.setattr(httpx, "get", _get)

    result = await ai_interaction.do_generate_image("draw a chair\ndall-e-3")

    assert result["error"] == (
        "Image API returned unsafe image URL: "
        "link-local address blocked (SSRF metadata risk): 169.254.169.254"
    )
    assert events == [("check", unsafe_url, False)]


_PROVIDER_PATHS = ["ai_dashscope", "mcp_dashscope", "mcp_openai"]


async def _run_provider_image(monkeypatch, tmp_path, provider_path, image_url, *, b64_image=None, posted=None, public_url=""):
    import mcp_servers.image_gen_server as image_server
    import src.database as database

    _patch_generation(
        monkeypatch, image_url,
        dashscope=provider_path.endswith("dashscope"),
        b64_image=b64_image, posted=posted, public_url=public_url,
    )
    monkeypatch.setattr(ai_interaction, "GENERATED_IMAGES_DIR", str(tmp_path))
    monkeypatch.setattr(image_server, "GENERATED_IMAGES_DIR", str(tmp_path))
    monkeypatch.setattr(database, "SessionLocal", MagicMock())
    if provider_path == "ai_dashscope":
        return await ai_interaction.do_generate_image("draw a chair\nqwen-image")
    result = await image_server.call_tool("generate_image", {"prompt": "draw a chair", "model": "qwen-image"})
    text = result[0].text
    if text.startswith("Error:"):
        return {"error": text}
    return {"image_url": text.split("Direct link: ", 1)[1].splitlines()[0], "output": text}


@pytest.mark.parametrize("provider_path", _PROVIDER_PATHS)
async def test_provider_image_rejects_metadata_url_before_fetch(monkeypatch, tmp_path, provider_path):
    import httpx

    opened = []

    def unexpected_fetch(*args, **kwargs):
        opened.append(True)
        raise AssertionError("metadata URL must be rejected before opening a client")

    monkeypatch.setattr(httpx, "Client", unexpected_fetch)
    result = await _run_provider_image(
        monkeypatch, tmp_path, provider_path, "http://169.254.169.254/latest/meta-data",
    )

    assert "could not be downloaded safely" in result["error"]
    assert "image_url" not in result
    assert not list(tmp_path.iterdir())
    assert opened == []


@pytest.mark.parametrize("provider_path", _PROVIDER_PATHS)
async def test_provider_image_rejects_redirect_to_metadata(monkeypatch, tmp_path, provider_path):
    import httpx
    import src.outbound_fetch as outbound_fetch

    requested = []

    def redirect(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

    monkeypatch.setattr(outbound_fetch, "_PinnedTransport", lambda _ip: httpx.MockTransport(redirect))
    result = await _run_provider_image(
        monkeypatch, tmp_path, provider_path, "https://8.8.8.8/generated.png",
    )

    assert "could not be downloaded safely" in result["error"]
    assert requested == ["https://8.8.8.8/generated.png"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("provider_path", _PROVIDER_PATHS)
async def test_provider_image_rejects_truncated_download(monkeypatch, tmp_path, provider_path):
    import httpx
    import src.constants as constants
    import src.outbound_fetch as outbound_fetch

    monkeypatch.setattr(constants, "WEB_FETCH_HARD_MAX_BYTES", 8)
    monkeypatch.setattr(
        outbound_fetch, "_PinnedTransport",
        lambda _ip: httpx.MockTransport(lambda _request: httpx.Response(200, content=b"x" * 16)),
    )
    result = await _run_provider_image(
        monkeypatch, tmp_path, provider_path, "https://8.8.8.8/generated.png",
    )

    assert "could not be downloaded safely" in result["error"]
    assert "image_url" not in result
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("provider_path", _PROVIDER_PATHS)
async def test_provider_image_saves_complete_download_off_event_loop(monkeypatch, tmp_path, provider_path):
    import httpx
    import src.outbound_fetch as outbound_fetch

    loop_thread = threading.get_ident()
    fetch_threads = []
    pinned_ips = []
    posted = []
    image_bytes = b"complete-image"

    def download(request):
        fetch_threads.append(threading.get_ident())
        assert "authorization" not in request.headers
        return httpx.Response(200, content=image_bytes)

    def transport(ip):
        pinned_ips.append(ip)
        return httpx.MockTransport(download)

    monkeypatch.setattr(outbound_fetch, "_PinnedTransport", transport)
    result = await _run_provider_image(
        monkeypatch, tmp_path, provider_path, "https://8.8.8.8/generated.png", posted=posted,
    )

    assert "error" not in result
    assert result["image_url"].startswith("/api/generated-image/")
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [image_bytes]
    assert fetch_threads and all(thread != loop_thread for thread in fetch_threads)
    assert pinned_ips == [ipaddress.ip_address("8.8.8.8")]
    if provider_path.endswith("dashscope"):
        url, payload = posted[0]
        assert url.endswith("/api/v1/services/aigc/multimodal-generation/generation")
        assert payload["input"]["messages"][0]["content"] == [{"text": "draw a chair"}]
        assert payload["parameters"]["size"] == "1024*1024"
    if provider_path == "ai_dashscope":
        assert result["image_prompt"] == "draw a chair"
        assert result["image_model"] == "qwen-image"
        assert result["image_size"] == "1024x1024"
    else:
        assert result["output"].startswith("Generated image for: draw a chair\nDirect link:")


@pytest.mark.parametrize("provider_path", _PROVIDER_PATHS)
async def test_provider_base64_image_is_saved_without_refetch(monkeypatch, tmp_path, provider_path):
    import src.outbound_fetch as outbound_fetch

    image_bytes = b"base64-image"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("saved base64 image must not be downloaded again")

    monkeypatch.setattr(outbound_fetch, "_get_public_url", unexpected_fetch)
    result = await _run_provider_image(
        monkeypatch, tmp_path, provider_path, "data:image/png;base64," + encoded,
        b64_image=encoded, public_url="http://127.0.0.1:7000",
    )

    assert "error" not in result
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [image_bytes]
    if provider_path.startswith("mcp"):
        assert result["image_url"].startswith("http://127.0.0.1:7000/api/generated-image/")
