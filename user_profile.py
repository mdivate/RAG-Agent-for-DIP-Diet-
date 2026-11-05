# user_profile.py — User profile data model and session management

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# Mapping from allergen display name to search keywords (root forms)
_ALLERGEN_KEYWORDS: dict = {
    "Peanuts": ["peanut"],
    "Tree Nuts": ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "macadamia", "tree nut"],
    "Dairy / Milk": ["milk", "dairy", "cream", "butter", "cheese", "yogurt", "yoghurt", "lactose", "whey", "casein"],
    "Eggs": ["egg", "eggs", "mayonnaise", "mayo"],
    "Wheat / Gluten": ["wheat", "gluten", "flour", "bread", "pasta", "barley", "rye", "semolina"],
    "Soy": ["soy", "soya", "tofu", "tempeh", "edamame", "miso"],
    "Fish": ["fish", "salmon", "tuna", "cod", "tilapia", "anchov", "sardine", "halibut", "bass", "trout"],
    "Shellfish": ["shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel", "oyster", "shellfish"],
    "Sesame": ["sesame", "tahini"],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """All personal and dietary details collected from the user."""

    # Basic demographics
    name: str = ""
    age: int = 0
    location: str = ""

    # Dietary identity
    food_preference: str = "No preference"   # Vegetarian / Vegan / Non-Veg / No preference
    dietary_restrictions: List[str] = field(default_factory=list)  # e.g. Gluten-Free, Keto
    allergies: List[str] = field(default_factory=list)             # e.g. Peanuts, Dairy

    # Culinary context
    cuisine_preference: List[str] = field(default_factory=list)    # e.g. Indian, Italian
    available_ingredients: str = ""                                 # Free-text list
    cooking_skill_level: str = "Intermediate"                      # Beginner / Intermediate / Advanced
    cooking_time_preference: str = "Any"                           # Quick (<30 min) / Medium / Any

    # Health context
    health_goals: List[str] = field(default_factory=list)          # e.g. Weight Loss, High Protein
    calorie_target: Optional[int] = None                           # kcal per meal (optional)

    # Profile completion flag
    profile_complete: bool = False

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def to_prompt_context(self) -> str:
        """
        Render the profile as a concise natural-language context block
        suitable for injection into LLM prompts.
        """
        lines = []
        if self.name:
            lines.append(f"User Name: {self.name}")
        if self.age:
            lines.append(f"Age: {self.age}")
        if self.location:
            lines.append(f"Location: {self.location}")
        lines.append(f"Food Preference: {self.food_preference}")
        if self.dietary_restrictions:
            lines.append(f"Dietary Restrictions: {', '.join(self.dietary_restrictions)}")
        if self.allergies:
            lines.append(f"Allergies: {', '.join(self.allergies)}")
        if self.cuisine_preference:
            lines.append(f"Preferred Cuisines: {', '.join(self.cuisine_preference)}")
        if self.available_ingredients:
            lines.append(f"Available Ingredients: {self.available_ingredients}")
        lines.append(f"Cooking Skill Level: {self.cooking_skill_level}")
        lines.append(f"Preferred Cooking Time: {self.cooking_time_preference}")
        if self.health_goals:
            lines.append(f"Health Goals: {', '.join(self.health_goals)}")
        if self.calorie_target:
            lines.append(f"Calorie Target per Meal: ~{self.calorie_target} kcal")
        return "\n".join(lines)

    def get_allergy_warning(self, recipe_text: str) -> List[str]:
        """
        Scan recipe_text for known allergens and return a list of warnings.
        Uses root-keyword matching so 'Peanuts' catches 'peanut sauce', etc.
        This is a keyword scan — always advise verifying manually.
        """
        warnings: List[str] = []
        recipe_lower = recipe_text.lower()
        for allergen in self.allergies:
            # Use expanded keyword list if available, else fall back to the display name root
            keywords = _ALLERGEN_KEYWORDS.get(allergen, [allergen.lower().split(" ")[0]])
            for kw in keywords:
                if kw in recipe_lower:
                    warnings.append(allergen)
                    break  # one match per allergen is enough
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Allowed option sets (used by the UI to populate select boxes)
# ---------------------------------------------------------------------------

FOOD_PREFERENCE_OPTIONS = [
    "No preference",
    "Vegetarian",
    "Vegan",
    "Non-Vegetarian",
    "Pescatarian",
]

DIETARY_RESTRICTION_OPTIONS = [
    "Gluten-Free",
    "Dairy-Free",
    "Nut-Free",
    "Keto",
    "Paleo",
    "Low-Sodium",
    "Low-Sugar",
    "Halal",
    "Kosher",
]

COMMON_ALLERGEN_OPTIONS = [
    "Peanuts",
    "Tree Nuts",
    "Dairy / Milk",
    "Eggs",
    "Wheat / Gluten",
    "Soy",
    "Fish",
    "Shellfish",
    "Sesame",
]

CUISINE_OPTIONS = [
    "Indian",
    "Italian",
    "Chinese",
    "Mexican",
    "Mediterranean",
    "Japanese",
    "Thai",
    "American",
    "French",
    "Middle Eastern",
    "Korean",
    "Greek",
    "Spanish",
    "Vietnamese",
    "Global / Mixed",
]

SKILL_LEVEL_OPTIONS = ["Beginner", "Intermediate", "Advanced"]

COOKING_TIME_OPTIONS = [
    "Quick (<30 min)",
    "Medium (30–60 min)",
    "Slow (60+ min)",
    "Any",
]

HEALTH_GOAL_OPTIONS = [
    "Weight Loss",
    "Muscle Gain / High Protein",
    "Heart Health / Low Fat",
    "Diabetes-Friendly / Low Sugar",
    "Energy Boost",
    "Immunity Boosting",
    "General Balanced Diet",
]
