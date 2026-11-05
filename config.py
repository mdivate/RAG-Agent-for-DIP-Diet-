# config.py — Central configuration for the RAG Recipe Generator

IBM_API_KEY = "qQfNCZEYbrhvQgGnSQz93B0iI1NZikTPCEwVxmHFzwec"
IBM_PROJECT_ID = "d0ee319c-6bad-444f-ab33-175b49c642ee"
IBM_ENDPOINT_URL = "https://au-syd.ml.cloud.ibm.com"

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
