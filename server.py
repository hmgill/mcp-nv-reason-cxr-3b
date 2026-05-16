"""
server.py — cxr-mcp
=====================
FastMCP server exposing NV-Reason-CXR-3B chest X-ray reasoning as MCP
tools, deployed via Prefect Horizon.

Image preprocessing runs locally (decode + validate); GPU inference is
dispatched to a Modal serverless endpoint so Horizon needs no GPU or
model weights.

Required environment variables:
    MODAL_ENDPOINT_URL  Full Modal endpoint base URL,
                        e.g. https://<workspace>--cxr-reasoning-fastapi-app.modal.run

Optional environment variables:
    FASTMCP_DOCKET_URL  rediss://<host>:<port>  Redis for background tasks

Tools:
    analyze_cxr(image_b64, image_id, prompt)  → findings + structured reasoning
    reason_cxr(image_b64, image_id, prompt)   → targeted clinical reasoning
    health()                                  → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os

import requests
from fastmcp import FastMCP
from PIL import Image

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")

if not MODAL_ENDPOINT_URL:
    logger.warning("MODAL_ENDPOINT_URL is not set — inference calls will fail.")


# ---------------------------------------------------------------------------
# Modal client
# ---------------------------------------------------------------------------

def _modal_dispatch(route: str, image_id: str, image_b64: str, prompt: str) -> dict:
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    url = f"{MODAL_ENDPOINT_URL}{route}"
    logger.info(f"[{image_id}] Dispatching to Modal: {url}")

    resp = requests.post(
        url,
        json={"image_id": image_id, "image_b64": image_b64, "prompt": prompt},
        timeout=180,  # CXR reasoning can take longer than AV segmentation
    )
    resp.raise_for_status()
    output = resp.json()

    if not output.get("success"):
        raise RuntimeError(f"Modal inference failed: {output.get('error')}")

    return output


# ---------------------------------------------------------------------------
# Preprocessing — runs locally, no GPU needed
# ---------------------------------------------------------------------------

def _preprocess_image(image_b64: str, image_id: str) -> str:
    """
    Decode, validate, and re-encode the image as PNG.
    Ensures the Modal worker always receives a clean RGB PNG regardless
    of the input format (JPEG, PNG, DICOM-exported PNG, etc.).
    """
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"[{image_id}] Could not decode image: {e}")

    # Sanity-check dimensions — reject obviously non-CXR tiny images
    w, h = img.size
    if w < 64 or h < 64:
        raise ValueError(
            f"[{image_id}] Image too small ({w}x{h}). "
            "Expected a chest X-ray of reasonable resolution."
        )

    logger.info(f"[{image_id}] Image validated: {w}x{h} RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("cxr-mcp")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_cxr(image_b64: str, image_id: str, prompt: str = "Find abnormalities and support devices.") -> str:
    """
    Analyze a chest X-ray image using NV-Reason-CXR-3B.

    Performs general radiological analysis with step-by-step reasoning.
    The model returns structured output with a thinking chain and a
    concise answer. Suitable for abnormality detection, device identification,
    and general findings summarization.

    This model is for research and educational purposes only. Outputs
    should not be used for clinical diagnosis or treatment decisions.

    Args:
        image_b64:  Base64-encoded chest X-ray image (JPEG or PNG).
        image_id:   Identifier for this image (used for logging).
        prompt:     Clinical question or analysis request. Defaults to
                    general abnormality + support device detection.

    Returns:
        JSON with thinking chain, concise answer, and raw model output.
    """
    from datetime import datetime, timezone

    try:
        clean_b64 = _preprocess_image(image_b64, image_id)
        output = _modal_dispatch("/analyze_cxr", image_id, clean_b64, prompt)

        payload = json.dumps({
            "success":    True,
            "image_id":   image_id,
            "prompt":     prompt,
            "thinking":   output.get("thinking", ""),
            "answer":     output.get("answer", ""),
            "raw_text":   output.get("raw_text", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "For research and educational use only. "
                "Not for clinical diagnosis or treatment decisions."
            ),
        })

        logger.info(f"analyze_cxr: {image_id}  payload={len(payload)/1024:.1f}KB")
        return payload

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"analyze_cxr failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def reason_cxr(image_b64: str, image_id: str, prompt: str) -> str:
    """
    Perform targeted clinical reasoning on a chest X-ray image.

    Sends a specific clinical question to NV-Reason-CXR-3B and returns
    a structured reasoning chain with a focused answer. Use this when
    you have a specific hypothesis or finding to evaluate (e.g. "Is there
    evidence of pneumothorax?", "Describe the cardiac silhouette.").

    This model is for research and educational purposes only. Outputs
    should not be used for clinical diagnosis or treatment decisions.

    Args:
        image_b64:  Base64-encoded chest X-ray image (JPEG or PNG).
        image_id:   Identifier for this image (used for logging).
        prompt:     Specific clinical question or reasoning request.

    Returns:
        JSON with thinking chain, focused answer, and raw model output.
    """
    from datetime import datetime, timezone

    try:
        clean_b64 = _preprocess_image(image_b64, image_id)
        output = _modal_dispatch("/reason_cxr", image_id, clean_b64, prompt)

        payload = json.dumps({
            "success":    True,
            "image_id":   image_id,
            "prompt":     prompt,
            "thinking":   output.get("thinking", ""),
            "answer":     output.get("answer", ""),
            "raw_text":   output.get("raw_text", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "For research and educational use only. "
                "Not for clinical diagnosis or treatment decisions."
            ),
        })

        logger.info(f"reason_cxr: {image_id}  payload={len(payload)/1024:.1f}KB")
        return payload

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"reason_cxr failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "cxr-mcp",
        "modal": {
            "endpoint_url": MODAL_ENDPOINT_URL or "(not set)",
            "configured":   bool(MODAL_ENDPOINT_URL),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
