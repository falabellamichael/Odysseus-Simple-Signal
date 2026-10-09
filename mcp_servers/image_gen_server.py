"""
image_gen_server.py

MCP server exposing image generation via OpenAI-compatible APIs.
"""

import asyncio
import base64
import sys
import uuid
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import GENERATED_IMAGES_DIR

server = Server("image_gen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_image",
            description="Generate an image using an image-capable model (e.g. gpt-image-1)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description prompt"},
                    "model": {"type": "string", "description": "Model name (auto-detects if omitted)"},
                    "size": {"type": "string", "description": "Image size (default 1024x1024)"},
                    "quality": {"type": "string", "description": "Quality: low, medium, high, auto (default medium)"},
                },
                "required": ["prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "generate_image":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    prompt = arguments.get("prompt", "")
    model_spec = arguments.get("model", "")
    size = arguments.get("size", "1024x1024")
    quality = arguments.get("quality", "medium")

    if not prompt:
        return [TextContent(type="text", text="Error: Image prompt is required")]

    try:
        import httpx
        from src.settings import load_settings, get_setting
        from src.ai_interaction import _download_generated_image_url, _resolve_model

        if not get_setting("image_gen_enabled", True):
            return [TextContent(type="text", text="Error: Image generation is disabled by the administrator.")]

        _settings = load_settings()

        if not model_spec:
            model_spec = _settings.get("image_model", "")
        if quality == "medium" and _settings.get("image_quality"):
            quality = _settings["image_quality"]

        # Auto-detect best available image model
        if not model_spec:
            for candidate in ("gpt-image-1.5", "gpt-image-1", "dall-e-3"):
                try:
                    await asyncio.to_thread(_resolve_model, candidate)
                    model_spec = candidate
                    break
                except ValueError:
                    continue
            if not model_spec:
                return [TextContent(type="text", text="Error: No image model found. Configure one in Admin.")]

        try:
            url, model_id, headers = await asyncio.to_thread(_resolve_model, model_spec, model_type="image")
        except ValueError:
            _lower_model_spec = model_spec.lower()
            if not any(_name in _lower_model_spec for _name in ("gpt-image", "dall-e")):
                raise
            url, model_id, headers = await asyncio.to_thread(_resolve_model, model_spec)

        is_gpt_image = "gpt-image" in model_id.lower()
        base_url = url.replace("/chat/completions", "").replace("/v1/messages", "").rstrip("/")
        base_lower = base_url.lower()
        is_dashscope = "aliyuncs.com" in base_lower or "dashscope" in base_lower

        if is_dashscope:
            from urllib.parse import urlsplit, urlunsplit
            parsed = urlsplit(base_url)
            path = parsed.path.rstrip("/")
            lower_path = path.lower()
            if "/compatible-mode/v1" in lower_path:
                path = path[:lower_path.index("/compatible-mode/v1")] + "/api/v1"
            elif "/api/v1" in lower_path:
                path = path[:lower_path.index("/api/v1") + len("/api/v1")]
            elif lower_path.endswith("/v1"):
                path = path[:-3].rstrip("/") + "/api/v1"
            else:
                path = path + "/api/v1" if path else "/api/v1"
            images_url = urlunsplit((
                parsed.scheme, parsed.netloc,
                path + "/services/aigc/multimodal-generation/generation", "", ""
            ))
            dash_size = size.replace("x", "*")
            payload = {
                "model": model_id,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }]
                },
                "parameters": {
                    "size": dash_size,
                    "n": 1,
                },
            }
        else:
            images_url = base_url + "/images/generations"

            valid_gpt_sizes = {"1024x1024", "1024x1536", "1536x1024", "auto"}
            valid_dalle3_sizes = {"1024x1024", "1024x1792", "1792x1024"}
            if is_gpt_image and size not in valid_gpt_sizes:
                size = "1024x1024"
            elif not is_gpt_image and size not in valid_dalle3_sizes:
                size = "1024x1024"

            payload = {"model": model_id, "prompt": prompt, "n": 1, "size": size}
        if is_gpt_image:
            payload["quality"] = quality if quality in ("low", "medium", "high", "auto") else "medium"

        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)) as client:
            resp = await client.post(images_url, json=payload, headers=headers)

            if resp.status_code != 200:
                error_text = resp.text[:500]
                try:
                    err_json = resp.json()
                    error_text = err_json.get("error", {}).get("message", error_text) if isinstance(err_json.get("error"), dict) else str(err_json.get("error", error_text))
                except Exception:
                    pass
                return [TextContent(type="text", text=f"Error: Image generation failed ({resp.status_code}): {error_text}")]

            data = resp.json()
            remote_image_url = None

            if is_dashscope:
                # DashScope returns: output.choices[0].message.content[0].image
                # The image field can be an HTTP URL or a base64 data URI.
                image_url = None
                try:
                    output = data.get("output", {})
                    choices = output.get("choices", [])
                    if choices:
                        msg_content = choices[0].get("message", {}).get("content", [])
                        if msg_content:
                            image_url = msg_content[0].get("image", "")
                except Exception:
                    pass
                if not image_url:
                    import json as _json
                    def _find_image(node, depth=0):
                        if node is None or depth > 5:
                            return None
                        if isinstance(node, str):
                            s = node.strip()
                            if s.startswith("http://") or s.startswith("https://") or s.startswith("data:image/"):
                                return s
                            return None
                        if isinstance(node, list):
                            for item in node:
                                r = _find_image(item, depth + 1)
                                if r:
                                    return r
                            return None
                        if isinstance(node, dict):
                            for k in ("image", "url", "b64_json", "image_url"):
                                if k in node:
                                    r = _find_image(node[k], depth + 1)
                                    if r:
                                        return r
                            return None
                        return None
                    image_url = _find_image(data) or ""
                if not image_url:
                    return [TextContent(type="text", text="Error: DashScope returned no image URL")]
                if image_url.startswith(("http://", "https://")):
                    remote_image_url = image_url

                # Handle base64 data URIs from DashScope
                _pub_base = (get_setting("app_public_url", "") or "").rstrip("/")
                if image_url.startswith("data:image/"):
                    try:
                        header, b64_data = image_url.split(",", 1)
                        raw_bytes = base64.b64decode(b64_data)
                        img_dir = Path(GENERATED_IMAGES_DIR)
                        img_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{uuid.uuid4().hex[:12]}.png"
                        img_path = img_dir / filename
                        img_path.write_bytes(raw_bytes)
                        image_url = f"{_pub_base}/api/generated-image/{filename}"
                        try:
                            from src.database import SessionLocal, GalleryImage
                            db = SessionLocal()
                            db.add(GalleryImage(
                                id=str(uuid.uuid4()),
                                filename=filename,
                                prompt=prompt,
                                model=model_id,
                                size=size,
                                quality=payload.get("quality", "medium"),
                            ))
                            db.commit()
                            db.close()
                        except Exception:
                            pass
                    except Exception:
                        pass  # Keep data URI as fallback
            else:
                images = data.get("data", [])
                if not images:
                    return [TextContent(type="text", text="Error: No images returned from API")]

                img = images[0]
                image_url = None
                # Prefix the instance's public base URL (existing app_public_url setting) so the
                # link is fully-qualified and clickable when the model echoes it. Empty = relative
                # same-origin path (unchanged default).
                _pub_base = (get_setting("app_public_url", "") or "").rstrip("/")

                if img.get("b64_json"):
                    img_dir = Path(GENERATED_IMAGES_DIR)
                    img_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{uuid.uuid4().hex[:12]}.png"
                    img_path = img_dir / filename
                    img_path.write_bytes(base64.b64decode(img["b64_json"]))
                    image_url = f"{_pub_base}/api/generated-image/{filename}"

                    # Save to gallery
                    try:
                        from src.database import SessionLocal, GalleryImage
                        db = SessionLocal()
                        db.add(GalleryImage(
                            id=str(uuid.uuid4()),
                            filename=filename,
                            prompt=prompt,
                            model=model_id,
                            size=size,
                            quality=payload.get("quality", "medium"),
                        ))
                        db.commit()
                        db.close()
                    except Exception:
                        pass

                elif img.get("url"):
                    image_url = img["url"]
                    remote_image_url = image_url
                else:
                    return [TextContent(type="text", text="Error: Unexpected image API response format")]

            # Download remote URLs (DashScope returns temporary URLs) to local storage
            if remote_image_url:
                try:
                    dl = await _download_generated_image_url(remote_image_url)
                    if dl.status_code == 200:
                        img_dir = Path(GENERATED_IMAGES_DIR)
                        img_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{uuid.uuid4().hex[:12]}.png"
                        img_path = img_dir / filename
                        img_path.write_bytes(dl.content)
                        _pub_base = (get_setting("app_public_url", "") or "").rstrip("/")
                        image_url = f"{_pub_base}/api/generated-image/{filename}"
                        try:
                            from src.database import SessionLocal, GalleryImage
                            db = SessionLocal()
                            db.add(GalleryImage(
                                id=str(uuid.uuid4()),
                                filename=filename,
                                prompt=prompt,
                                model=model_id,
                                size=size,
                                quality=payload.get("quality", "medium"),
                            ))
                            db.commit()
                            db.close()
                        except Exception:
                            pass
                except Exception:
                    return [TextContent(
                        type="text",
                        text="Error: Generated image URL could not be downloaded safely.",
                    )]

            # "Direct link:" rather than an "image_url:" label — small models copied the
            # label token ("image_url") into the link href, producing a broken link.
            result = (
                f"Generated image for: {prompt[:100]}\n"
                f"Direct link: {image_url}\n"
                f"model: {model_id}\nsize: {size}"
            )
            return [TextContent(type="text", text=result)]

    except httpx.TimeoutException:
        return [TextContent(type="text", text="Error: Image generation timed out (300s)")]
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
