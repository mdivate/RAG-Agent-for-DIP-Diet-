# llm_client.py — IBM watsonx.ai / Granite LLM integration

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


# ---------------------------------------------------------------------------
# Singleton LLM client
# ---------------------------------------------------------------------------

_LLM_CLIENT: Optional[ModelInference] = None
_LLM_ERROR: Optional[str] = None   # stores the last initialisation error
_LLM_MODEL_ID: Optional[str] = None  # tracks which model_id the client was built with


def _cfg():
    """Always reload config so we pick up the current GRANITE_MODEL_ID value."""
    import config  # noqa: PLC0415
    importlib.reload(config)
    return config


def _get_llm() -> ModelInference:
    """
    Lazily initialise and return the IBM Granite ModelInference client.
    Automatically rebuilds the client if the model ID in config has changed.
    Raises RuntimeError with a user-friendly message if initialisation fails.
    """
    global _LLM_CLIENT, _LLM_ERROR, _LLM_MODEL_ID

    cfg = _cfg()

    # Invalidate the cached client if the model ID has changed since last build
    if _LLM_CLIENT is not None and _LLM_MODEL_ID != cfg.GRANITE_MODEL_ID:
        _LLM_CLIENT = None
        _LLM_ERROR = None
        _LLM_MODEL_ID = None

    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    # If a previous attempt already failed for the *same* model, surface cached error
    if _LLM_ERROR is not None:
        raise RuntimeError(_LLM_ERROR)

    try:
        credentials = Credentials(
            url=cfg.get_ibm_endpoint_url(),
            api_key=cfg.get_ibm_api_key(),
        )
        _LLM_CLIENT = ModelInference(
            model_id=cfg.GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=cfg.get_ibm_project_id(),
            params={
                GenParams.MAX_NEW_TOKENS: cfg.LLM_PARAMS["max_new_tokens"],
                GenParams.MIN_NEW_TOKENS: cfg.LLM_PARAMS["min_new_tokens"],
                GenParams.TEMPERATURE: cfg.LLM_PARAMS["temperature"],
                GenParams.TOP_P: cfg.LLM_PARAMS["top_p"],
                GenParams.TOP_K: cfg.LLM_PARAMS["top_k"],
                GenParams.REPETITION_PENALTY: cfg.LLM_PARAMS["repetition_penalty"],
            },
            # validate=False skips GET /foundation_model_specs on init
            validate=False,
        )
        _LLM_MODEL_ID = cfg.GRANITE_MODEL_ID
        _LLM_ERROR = None
        return _LLM_CLIENT

    except Exception as exc:
        _LLM_ERROR = _friendly_error(exc)
        raise RuntimeError(_LLM_ERROR) from exc


def reset_llm_client() -> None:
    """
    Clear the cached client and error so the next call re-attempts initialisation.
    Call this after the user updates their IBM Cloud configuration.
    """
    global _LLM_CLIENT, _LLM_ERROR
    _LLM_CLIENT = None
    _LLM_ERROR = None


def _extract_chat_response(response: dict) -> str:
    """Extract the assistant text from a chat() API response."""
    choices = response.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    # Fallback: try generate() response shape
    return response.get("results", [{}])[0].get("generated_text", "").strip()


def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a user prompt to the LLM and return the generated text.
    Uses the chat() API (/ml/v1/text/chat — non-deprecated).
    Returns a human-readable [LLM Error] string on failure.
    """
    try:
        llm = _get_llm()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            params={GenParams.MAX_NEW_TOKENS: max_tokens},
        )
        return _extract_chat_response(response)
    except Exception as exc:
        return f"[LLM Error] {exc}"


def generate_chat(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Send a system + user message pair to the LLM via the chat API.
    Preferred over generate_text when a distinct system role is needed.
    Returns a human-readable [LLM Error] string on failure.
    """
    try:
        llm = _get_llm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = llm.chat(
            messages=messages,
            params={GenParams.MAX_NEW_TOKENS: max_tokens},
        )
        return _extract_chat_response(response)
    except Exception as exc:
        return f"[LLM Error] {exc}"


def test_connection() -> Dict[str, Any]:
    """
    Smoke-test the IBM watsonx.ai connection.
    Returns a dict with ``success`` (bool) and ``message`` (str).
    """
    # Reset cached error so a fresh attempt is made each time the user clicks
    reset_llm_client()
    try:
        result = generate_text(
            "Hello! Respond with exactly: IBM Granite is connected.",
            max_tokens=20,
        )
        if result.startswith("[LLM Error]"):
            return {"success": False, "message": result}
        return {"success": True, "message": result}
    except Exception as exc:
        return {"success": False, "message": _friendly_error(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _friendly_error(exc: Exception) -> str:
    """Convert a raw IBM SDK exception into a concise, actionable message."""
    msg = str(exc)

    if "no_associated_service_instance_error" in msg:
        return (
            "❌ IBM Cloud Setup Required: The project is not linked to a "
            "Watson Machine Learning instance.\n\n"
            "Fix: Go to IBM Cloud → your watsonx.ai project → "
            "Manage → Services & integrations → Associate service → "
            "select or create a Watson Machine Learning instance."
        )
    if "403" in msg or "Unauthorized" in msg or "forbidden" in msg.lower():
        return (
            "❌ Access Denied (403): Check that your API key is correct and "
            "has access to the project."
        )
    if "401" in msg:
        return (
            "❌ Authentication Failed (401): Your IBM API key may be invalid or expired. "
            "Generate a new key at cloud.ibm.com → Manage → Access (IAM) → API keys."
        )
    if "ConnectionError" in msg or "timeout" in msg.lower():
        return (
            "❌ Network Error: Cannot reach IBM watsonx.ai. "
            "Check your internet connection and that the endpoint URL is correct."
        )
    return f"❌ IBM watsonx.ai Error: {msg}"
