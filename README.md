# 🍽️ RAG Agent for DIP Diet — AI Recipe Generator

> **Personalized recipes powered by IBM Granite LLM & Retrieval-Augmented Generation (RAG)**

[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0f62fe?logo=ibm)](https://www.ibm.com/products/watsonx-ai)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-no-5-vweqdeo9nxhwkobvsk349l.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

A fully-featured web application that combines **Retrieval-Augmented Generation (RAG)** with **IBM Granite LLM** to deliver personalised, document-grounded recipe generation, nutrition analysis, and conversational cooking assistance.

Users can:
- Upload their own recipe books (PDF, DOCX, TXT) to build a personalised knowledge base
- Complete a detailed dietary profile (allergies, preferences, available ingredients, health goals)
- Generate complete recipes with ingredients, instructions, substitutions, and shopping lists
- Analyse the nutritional content of any recipe
- Adapt recipes to different dietary needs or cuisine styles
- Ask cooking questions via a conversational RAG-powered chat interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Application                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Recipe  │ │Nutrition │ │Personal- │ │  Q&A     │           │
│  │  Agent   │ │  Agent   │ │ ization  │ │  Agent   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └────────────┴────────────┴────────────┘                   │
│                          │                                        │
│              ┌───────────▼──────────┐                            │
│              │   Safety Guardrails   │                            │
│              └───────────┬──────────┘                            │
│                          │                                        │
│         ┌────────────────▼────────────────┐                      │
│         │      RAG Pipeline               │                      │
│         │  Document → Chunk → Embed       │                      │
│         │  → ChromaDB → Retrieve → Prompt │                      │
│         └────────────────┬────────────────┘                      │
│                          │                                        │
│         ┌────────────────▼────────────────┐                      │
│         │     IBM Granite 3 8B Instruct   │                      │
│         │     via IBM watsonx.ai API      │                      │
│         └─────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG-Agent-for-DIP-Diet/
├── app.py                    # Main Streamlit application
├── config.py                 # Central configuration (API params — keys via env vars)
├── document_processor.py     # PDF/DOCX/TXT parsing & text chunking
├── rag_pipeline.py           # Embeddings, ChromaDB vector store, retrieval
├── llm_client.py             # IBM watsonx.ai / Granite LLM integration
├── user_profile.py           # User data model & dietary profile
├── agents.py                 # All 4 agentic AI modules
├── sample_recipes.txt        # Sample recipe document for testing
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
└── README.md                 # This file
```

---

## 🌐 Live Demo

**👉 [https://project-no-5-vweqdeo9nxhwkobvsk349l.streamlit.app](https://project-no-5-vweqdeo9nxhwkobvsk349l.streamlit.app)**

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- An [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) account

### 2. Clone the Repository

```bash
git clone https://github.com/mdivate/RAG-Agent-for-DIP-Diet-.git
cd RAG-Agent-for-DIP-Diet-
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the sentence-transformer model (~90 MB). This is cached locally for subsequent runs.

### 4. Configure IBM watsonx.ai Credentials

Copy the environment variable template and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_ENDPOINT_URL=https://au-syd.ml.cloud.ibm.com
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🔑 IBM watsonx.ai Configuration

| Parameter | Description |
|-----------|-------------|
| `IBM_API_KEY` | Your IBM Cloud API key |
| `IBM_PROJECT_ID` | Your watsonx.ai project ID |
| `IBM_ENDPOINT_URL` | Regional endpoint (e.g. `https://au-syd.ml.cloud.ibm.com`) |
| Model | `meta-llama/llama-3-3-70b-instruct` |

> ⚠️ **Security:** Never commit your `.env` file or hardcode API keys. The `.gitignore` excludes `.env` automatically.

---

## 🎯 How to Use

### Step 1 — Complete Your Profile (👤 Profile tab)
Fill in your name, age, food preferences, allergies, dietary restrictions, available ingredients, and health goals. This personalises every recipe generated.

### Step 2 — Upload Recipe Documents (📚 Documents tab)
Upload PDF, DOCX, or TXT recipe files. The system chunks, embeds, and indexes them in ChromaDB. Use the included `sample_recipes.txt` to get started immediately.

### Step 3 — Generate Recipes (🧑‍🍳 Generate tab)
Describe what you want to cook. The system retrieves relevant document context and uses IBM Granite to generate a complete, personalised recipe with:
- Ingredients list with quantities
- Step-by-step instructions
- Ingredient substitutions
- Shopping list generation

### Nutrition Analysis (🥗 tab)
Paste any recipe to get estimated nutritional values, health alignment scores, and healthier ingredient swaps. Also includes a multi-day meal plan generator.

### Recipe Adaptation (✨ tab)
- **Adapt Recipe** — modify to match your dietary needs
- **Ingredient Substitutions** — find swaps for any ingredient
- **Cuisine Remix** — reimagine a recipe in a different cuisine style

### Cooking Chat (💬 tab)
Ask any cooking question. The AI uses your uploaded documents plus IBM Granite's knowledge to answer. Supports multi-turn conversation history.

---

## 🤖 Agentic AI Modules

| Agent | Responsibility |
|-------|---------------|
| `RecipeGenerationAgent` | Generates complete personalised recipes using RAG + user profile |
| `NutritionAnalysisAgent` | Nutritional breakdown, health goal alignment, meal planning |
| `PersonalizationAgent` | Recipe adaptation, ingredient substitution, cuisine reimagining |
| `ConversationalQAAgent` | Multi-turn Q&A with RAG context and conversation history |

---

## 🛡️ AI Ethics & Safety

| Principle | Implementation |
|-----------|---------------|
| **Helpful** | Detailed, actionable recipes and cooking guidance |
| **Accurate** | RAG grounds responses in real documents; sources cited |
| **Mindful (HAM)** | Inclusive tone, allergy detection, medical disclaimers |
| **Safe** | Keyword filter blocks harmful content requests |
| **Transparent** | Retrieved sources displayed; AI nature disclosed |
| **Private** | Documents stored locally; no PII retained between sessions |

---

## ⚠️ Disclaimers

> **Food Safety:** Always ensure ingredients are fresh and stored properly. Cook meat to safe internal temperatures.

> **Medical:** Nutritional information is approximate. Consult a qualified healthcare professional for medical dietary advice.

> **Allergies:** Always verify ingredient labels yourself. Cross-contamination can occur in shared kitchen environments.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `ibm-watsonx-ai` | IBM Granite LLM API client |
| `chromadb` | Local vector database |
| `sentence-transformers` | Text embedding model |
| `PyPDF2` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `langchain` | Text chunking utilities |
| `python-dotenv` | Load credentials from `.env` |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

*Built with ❤️ using IBM Granite LLM & watsonx.ai*
