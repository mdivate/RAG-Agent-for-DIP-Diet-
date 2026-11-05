# agents.py — All Agentic AI modules:
#   1. RecipeGenerationAgent
#   2. NutritionAnalysisAgent
#   3. PersonalizationAgent
#   4. ConversationalQAAgent
#
# Uses generate_chat(system, user) for proper system/user role separation
# with the Llama 3.3 instruct model via /ml/v1/text/chat.

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Supported output languages for recipe / nutrition generation
SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Spanish",
    "French",
    "German",
    "Arabic",
    "Portuguese",
    "Chinese (Simplified)",
    "Japanese",
    "Korean",
    "Italian",
    "Tamil",
    "Telugu",
    "Marathi",
    "Bengali",
]

from llm_client import generate_chat
from rag_pipeline import retrieve_relevant_chunks, build_context_string
from user_profile import UserProfile
from config import (
    FOOD_SAFETY_DISCLAIMER,
    MEDICAL_DISCLAIMER,
    ALLERGY_DISCLAIMER,
    TOP_K_RETRIEVAL,
)


# ===========================================================================
# Safety guardrail — keyword-based content filter
# ===========================================================================

_BLOCKED_KEYWORDS = [
    "poison", "toxic chemical", "bleach", "ammonia", "cyanide",
    "harmful substance", "drug", "narcotic",
]


def _safety_check(text: str) -> Optional[str]:
    """Return a refusal string if the text contains blocked keywords, else None."""
    lower = text.lower()
    for kw in _BLOCKED_KEYWORDS:
        if kw in lower:
            return (
                "I'm sorry, but I can't assist with that request as it may "
                "involve harmful or unsafe content. Please ask me about healthy, "
                "safe cooking and recipes."
            )
    return None


# ===========================================================================
# 1. Recipe Generation Agent
# ===========================================================================

class RecipeGenerationAgent:
    """Generates complete, personalised recipes using RAG context and user profile."""

    _SYSTEM = (
        "You are a professional chef and nutritionist AI assistant. "
        "You create detailed, safe, and delicious recipes tailored to the user's "
        "dietary preferences, allergies, and available ingredients. "
        "Always be helpful, respectful, and inclusive. "
        "Never suggest ingredients that conflict with stated allergies or dietary restrictions. "
        "Format all recipes with clear Markdown headings and bullet lists."
    )

    def generate(
        self,
        query: str,
        profile: UserProfile,
        use_documents: bool = True,
        top_k: int = TOP_K_RETRIEVAL,
        language: str = "English",
    ) -> Dict[str, Any]:
        """Generate a personalised recipe. Returns dict with recipe, chunks, warnings, disclaimer."""
        safety_block = _safety_check(query)
        if safety_block:
            return {"recipe": safety_block, "retrieved_chunks": [], "allergy_warnings": [], "disclaimer": ""}

        retrieved = []
        context_section = ""
        if use_documents:
            retrieved = retrieve_relevant_chunks(query, top_k=top_k)
            if retrieved:
                context_section = (
                    "## Reference Recipes from Knowledge Base\n"
                    + build_context_string(retrieved) + "\n\n"
                )

        user_ctx = profile.to_prompt_context() if profile.profile_complete else "No user profile provided."
        lang_instruction = f"7. Write the ENTIRE response in **{language}**." if language != "English" else ""

        user_msg = (
            f"## User Profile\n{user_ctx}\n\n"
            f"{context_section}"
            f"## Request\n{query}\n\n"
            "## Instructions\n"
            "Generate a complete recipe that:\n"
            "1. Suits the user's dietary preferences and restrictions.\n"
            "2. Avoids ALL listed allergens.\n"
            "3. Uses the available ingredients where possible.\n"
            "4. Is appropriate for the user's cooking skill level and time preference.\n"
            "5. Includes: Recipe Name, Servings & Prep/Cook Time, Ingredients List (with quantities), "
            "Step-by-step Instructions, Ingredient Substitutions, Tips & Variations.\n"
            "6. Format clearly with Markdown headings (##).\n"
            + (lang_instruction + "\n" if lang_instruction else "")
        )

        recipe_text = generate_chat(self._SYSTEM, user_msg, max_tokens=1200)

        allergy_warnings = profile.get_allergy_warning(recipe_text)
        disclaimer_parts = [FOOD_SAFETY_DISCLAIMER]
        if allergy_warnings:
            disclaimer_parts.append(
                f"**Allergen Alert:** The following allergens from your profile may appear in "
                f"this recipe — **{', '.join(allergy_warnings)}**. Please review carefully.\n\n"
                + ALLERGY_DISCLAIMER
            )

        return {
            "recipe": recipe_text,
            "retrieved_chunks": retrieved,
            "allergy_warnings": allergy_warnings,
            "disclaimer": "\n\n".join(disclaimer_parts),
        }

    def generate_shopping_list(self, recipe_text: str, profile: UserProfile) -> str:
        """Generate a formatted shopping list from an existing recipe."""
        available = profile.available_ingredients or "Not specified"
        system = "You are a helpful meal-planning assistant."
        user = (
            "Based on the recipe below, generate a concise shopping list grouped by category: "
            "Produce, Proteins, Dairy/Alternatives, Pantry, Spices.\n"
            f"Mark items the user already has (available: {available}) with ✅.\n\n"
            f"Recipe:\n{recipe_text[:1500]}\n\n"
            "Shopping List:"
        )
        return generate_chat(system, user, max_tokens=500)


# ===========================================================================
# 2. Nutrition Analysis Agent
# ===========================================================================

class NutritionAnalysisAgent:
    """Analyses the nutritional content of a recipe and provides health insights."""

    _SYSTEM = (
        "You are a registered dietitian AI assistant. "
        "You provide accurate, evidence-based nutritional analysis. "
        "Always include appropriate medical disclaimers and recommend consulting "
        "a healthcare professional for personalised medical nutrition therapy."
    )

    def analyse(self, recipe_text: str, profile: UserProfile, language: str = "English") -> str:
        """Return a detailed nutritional breakdown and health commentary."""
        safety_block = _safety_check(recipe_text)
        if safety_block:
            return safety_block

        health_goals = ", ".join(profile.health_goals) if profile.health_goals else "general healthy eating"
        calorie_note = f"Calorie target: ~{profile.calorie_target} kcal per meal. " if profile.calorie_target else ""
        lang_instruction = f"\n6. Write the ENTIRE response in **{language}**." if language != "English" else ""

        user = (
            f"Health goals: {health_goals}. {calorie_note}\n\n"
            f"Recipe to analyse:\n{recipe_text[:2000]}\n\n"
            "Provide the following analysis using Markdown formatting:\n"
            "1. **Estimated Nutritional Values per Serving** (Calories, Protein, Carbs, Fat, Fibre, Sodium)\n"
            "2. **Key Nutrients & Health Benefits**\n"
            "3. **Alignment with Health Goals** — rate and explain\n"
            "4. **Healthier Swaps** — 2–3 ingredient substitutions to improve nutrition\n"
            "5. **Important Notes** — for specific conditions (diabetes, heart health, etc.)"
            + lang_instruction
        )
        analysis = generate_chat(self._SYSTEM, user, max_tokens=900)
        return analysis + f"\n\n---\n{MEDICAL_DISCLAIMER}"

    def get_meal_plan_suggestion(self, profile: UserProfile, days: int = 3) -> str:
        """Generate a multi-day meal plan tailored to the user's profile."""
        system = "You are a meal planning dietitian AI assistant."
        user = (
            f"## User Profile\n{profile.to_prompt_context()}\n\n"
            f"Create a balanced {days}-day meal plan with Breakfast, Lunch, Dinner, and one Snack per day.\n"
            "For each meal include: meal name, approximate calories, and one key nutritional benefit.\n"
            "Ensure variety across days. Strictly follow dietary restrictions and avoid all allergens.\n"
            "Format as a Markdown table: | Day | Meal | Dish Name | Calories | Key Benefit |"
        )
        return generate_chat(system, user, max_tokens=900)


# ===========================================================================
# 3. Personalization Agent
# ===========================================================================

class PersonalizationAgent:
    """Adapts recipes to the user's constraints: substitutions, allergy removal, cuisine adaptation."""

    _SYSTEM = (
        "You are a culinary personalisation specialist. "
        "You modify recipes to perfectly fit individual dietary needs, "
        "preferences, and cultural considerations while preserving flavour and nutrition."
    )

    def adapt_recipe(self, recipe_text: str, profile: UserProfile, adaptation_request: str = "") -> str:
        """Adapt an existing recipe to fit the user's profile."""
        safety_block = _safety_check(adaptation_request)
        if safety_block:
            return safety_block

        extra = f"Additional request: {adaptation_request}\n\n" if adaptation_request else ""
        user = (
            f"## Original Recipe\n{recipe_text[:2000]}\n\n"
            f"## User Profile\n{profile.to_prompt_context()}\n\n"
            f"{extra}"
            "## Task\nAdapt the recipe to:\n"
            "1. Remove or substitute ingredients conflicting with the user's allergies and restrictions.\n"
            f"2. Align with the user's food preference: {profile.food_preference}.\n"
            "3. Use available ingredients where possible.\n"
            f"4. Match the user's cooking skill level: {profile.cooking_skill_level}.\n"
            "5. Apply any additional request listed above.\n\n"
            "Present the FULL adapted recipe with clear notes on what changed and why."
        )
        return generate_chat(self._SYSTEM, user, max_tokens=1000)

    def suggest_substitutions(self, ingredient: str, profile: UserProfile) -> str:
        """Suggest ingredient substitutions for a single ingredient."""
        allergies = ", ".join(profile.allergies) or "None"
        restrictions = ", ".join(profile.dietary_restrictions) or "None"
        user = (
            f"Suggest 3–5 substitutions for **{ingredient}** for a person with:\n"
            f"- Food preference: {profile.food_preference}\n"
            f"- Allergies: {allergies}\n"
            f"- Dietary restrictions: {restrictions}\n\n"
            "For each substitution explain: what it is, how to use it, and any quantity adjustments needed."
        )
        return generate_chat(self._SYSTEM, user, max_tokens=400)

    def adapt_for_cuisine(self, recipe_text: str, target_cuisine: str, profile: UserProfile) -> str:
        """Reimagine a recipe in the style of a different cuisine."""
        restrictions = ", ".join(profile.dietary_restrictions) or "None"
        allergies = ", ".join(profile.allergies) or "None"
        user = (
            f"## Original Recipe\n{recipe_text[:1500]}\n\n"
            f"Reimagine this recipe in **{target_cuisine}** cuisine style.\n"
            f"Maintain dietary restrictions ({restrictions}) and avoid allergens ({allergies}).\n"
            f"Use authentic {target_cuisine} ingredients and cooking techniques.\n\n"
            f"Provide the full adapted recipe."
        )
        return generate_chat(self._SYSTEM, user, max_tokens=900)


# ===========================================================================
# 4. Conversational Q&A Agent
# ===========================================================================

class ConversationalQAAgent:
    """Multi-turn cooking Q&A with RAG context and conversation history."""

    _SYSTEM = (
        "You are a knowledgeable, friendly, and helpful culinary AI assistant. "
        "Answer cooking questions clearly and practically. "
        "If information from uploaded recipe documents is available, prefer that. "
        "Always be respectful and encouraging. "
        "For medical or health claims, always advise consulting a qualified professional."
    )

    def answer(
        self,
        question: str,
        profile: Optional[UserProfile] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_documents: bool = True,
    ) -> Dict[str, Any]:
        """Answer a cooking question with RAG context. Returns dict with answer and retrieved_chunks."""
        safety_block = _safety_check(question)
        if safety_block:
            return {"answer": safety_block, "retrieved_chunks": []}

        retrieved = []
        context_section = ""
        if use_documents:
            retrieved = retrieve_relevant_chunks(question, top_k=3)
            if retrieved:
                context_section = (
                    "## Relevant Context from Uploaded Documents\n"
                    + build_context_string(retrieved) + "\n\n"
                )

        history_section = ""
        if conversation_history:
            lines = []
            for turn in conversation_history[-6:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                lines.append(f"{role}: {turn['content'][:300]}")
            history_section = "## Conversation History\n" + "\n".join(lines) + "\n\n"

        profile_section = ""
        if profile and profile.profile_complete:
            profile_section = f"## User Profile\n{profile.to_prompt_context()}\n\n"

        user_msg = (
            f"{context_section}"
            f"{profile_section}"
            f"{history_section}"
            f"## Question\n{question}"
        )

        answer_text = generate_chat(self._SYSTEM, user_msg, max_tokens=700)
        return {"answer": answer_text, "retrieved_chunks": retrieved}

    def explain_technique(self, technique: str) -> str:
        """Explain a specific cooking technique in detail."""
        system = "You are a professional culinary instructor with deep expertise in cooking techniques."
        user = (
            f"Explain the cooking technique: **{technique}**\n\n"
            "Include:\n"
            "1. What it is and when to use it\n"
            "2. Step-by-step instructions\n"
            "3. Common mistakes and how to avoid them\n"
            "4. Examples of dishes that use this technique"
        )
        return generate_chat(system, user, max_tokens=600)
