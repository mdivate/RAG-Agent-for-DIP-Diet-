# config.py — Central configuration for the RAG Recipe Generator
#
# Credentials are loaded from (in priority order):
#   1. Streamlit secrets  — set via App Settings → Secrets on share.streamlit.io
#   2. Environment variables / .env file (local development)
#
# st.secrets is accessed lazily (inside functions) so this module is safe
# to import before the Streamlit runtime initialises.

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set another way


def get_ibm_api_key() -> str:
    try:
        import streamlit as st
        return st.secrets.get("IBM_API_KEY", os.environ.get("IBM_API_KEY", ""))
    except Exception:
        return os.environ.get("IBM_API_KEY", "")


def get_ibm_project_id() -> str:
    try:
        import streamlit as st
        return st.secrets.get("IBM_PROJECT_ID", os.environ.get("IBM_PROJECT_ID", ""))
    except Exception:
        return os.environ.get("IBM_PROJECT_ID", "")


def get_ibm_endpoint_url() -> str:
    try:
        import streamlit as st
        return st.secrets.get("IBM_ENDPOINT_URL", os.environ.get("IBM_ENDPOINT_URL", "https://au-syd.ml.cloud.ibm.com"))
    except Exception:
        return os.environ.get("IBM_ENDPOINT_URL", "https://au-syd.ml.cloud.ibm.com")


# Module-level constants (non-secret) — safe to read at import time
IBM_ENDPOINT_URL = "https://au-syd.ml.cloud.ibm.com"  # default; overridden by get_ibm_endpoint_url()

# LLM Model — au-syd region (verified 2026-11)
# Only model supporting text_generation in au-syd:
#   meta-llama/llama-3-3-70b-instruct  ✅ text_generation, text_chat
# IBM Granite models in au-syd only support fine-tuning deployment, NOT text generation.
GRANITE_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
GRANITE_EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr-v2"

# LLM generation parameters
LLM_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# RAG / vector store settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 5
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "recipe_documents"

# Supported upload file types
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]

# Application metadata
APP_NAME = "🍽️ AI Recipe Generator"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Personalized recipes powered by IBM Granite & RAG"

# Safety & disclaimer text
FOOD_SAFETY_DISCLAIMER = (
    "⚠️ **Food Safety Notice:** Always ensure ingredients are fresh and stored properly. "
    "Cook meat and poultry to safe internal temperatures. When in doubt, consult a food safety professional."
)

MEDICAL_DISCLAIMER = (
    "⚕️ **Medical Disclaimer:** Nutritional information provided is approximate and for reference only. "
    "Individuals with specific medical conditions, allergies, or dietary needs should consult a qualified "
    "healthcare professional or registered dietitian before making dietary changes."
)

ALLERGY_DISCLAIMER = (
    "🚨 **Allergy Notice:** Although we attempt to flag allergens based on your profile, always verify "
    "all ingredient labels yourself. Cross-contamination can occur in shared kitchen environments."
)
