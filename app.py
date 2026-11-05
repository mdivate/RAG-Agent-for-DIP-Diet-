# app.py — Main Streamlit application: Document Q&A RAG Agent for Recipe Generator
# Powered by IBM Granite LLM & watsonx.ai

import streamlit as st
import time
from typing import Any, Dict, List

# ── Local modules ────────────────────────────────────────────────────────────
from config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    FOOD_SAFETY_DISCLAIMER,
    MEDICAL_DISCLAIMER,
    ALLERGY_DISCLAIMER,
    SUPPORTED_FILE_TYPES,
)
from document_processor import process_uploaded_file, get_document_summary
from rag_pipeline import (
    add_documents_to_index,
    get_index_stats,
    clear_index,
    list_indexed_sources,
    get_embedding_backend,
)
from user_profile import (
    UserProfile,
    FOOD_PREFERENCE_OPTIONS,
    DIETARY_RESTRICTION_OPTIONS,
    COMMON_ALLERGEN_OPTIONS,
    CUISINE_OPTIONS,
    SKILL_LEVEL_OPTIONS,
    COOKING_TIME_OPTIONS,
    HEALTH_GOAL_OPTIONS,
)
from agents import (
    RecipeGenerationAgent,
    NutritionAnalysisAgent,
    PersonalizationAgent,
    ConversationalQAAgent,
    SUPPORTED_LANGUAGES,
)
from llm_client import test_connection


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recipe Generator — IBM Granite",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #f9f6f2; }

    /* Card-like containers */
    .recipe-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0d8cc;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Accent headings */
    h1, h2, h3 { color: #2c3e50; }

    /* Disclaimer boxes */
    .disclaimer-box {
        background: #fff8e7;
        border-left: 4px solid #f39c12;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }

    /* Allergy alert */
    .allergy-alert {
        background: #fdecea;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
    }

    /* Success badge */
    .success-badge {
        background: #eafaf1;
        border-left: 4px solid #27ae60;
        padding: 0.6rem 1rem;
        border-radius: 4px;
    }

    /* IBM Granite badge */
    .ibm-badge {
        background: linear-gradient(135deg, #0f62fe, #4589ff);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Sidebar header */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #7f8c8d;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #e0d8cc;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "profile": UserProfile(),
        "chat_history": [],
        "last_recipe": "",
        "last_nutrition": "",
        "last_adapted_recipe": "",
        "uploaded_files_processed": [],
        "connection_tested": False,
        "connection_ok": False,
        "active_tab": "🏠 Home",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()

# ── Agent singletons ──────────────────────────────────────────────────────────
recipe_agent = RecipeGenerationAgent()
nutrition_agent = NutritionAnalysisAgent()
personalization_agent = PersonalizationAgent()
qa_agent = ConversationalQAAgent()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        # App branding
        st.markdown(
            f"""
            <div style="text-align:center; padding: 0.5rem 0 1rem;">
                <div style="font-size:2.5rem;">🍽️</div>
                <div style="font-size:1.2rem; font-weight:800; color:#2c3e50;">
                    AI Recipe Generator
                </div>
                <div style="font-size:0.8rem; color:#7f8c8d; margin-top:2px;">
                    {APP_TAGLINE}
                </div>
                <br/>
                <span class="ibm-badge">🤖 IBM Granite LLM</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Connection status ────────────────────────────────────────────────
        st.markdown('<div class="sidebar-header">⚡ IBM watsonx.ai Status</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("Test Connection", use_container_width=True, key="btn_test_conn"):
                with st.spinner("Connecting…"):
                    result = test_connection()
                    st.session_state.connection_ok = result["success"]
                    st.session_state.connection_tested = True
        with col2:
            if st.session_state.connection_tested:
                if st.session_state.connection_ok:
                    st.success("✅ Online")
                else:
                    st.error("❌ Offline")

        st.divider()

        # ── Knowledge base stats ─────────────────────────────────────────────
        st.markdown('<div class="sidebar-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
        stats = get_index_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Documents", stats["document_count"])
        col_b.metric("Chunks", stats["total_chunks"])

        _backend = get_embedding_backend()
        _backend_labels = {
            "sentence_transformers": ("🟢", "MiniLM (neural)"),
            "chromadb_default": ("🟡", "ChromaDB default"),
            "hash": ("🟠", "Hash fallback"),
            "none": ("⚪", "Not loaded"),
        }
        _icon, _label = _backend_labels.get(_backend, ("⚪", _backend))
        st.caption(f"Embedder: {_icon} {_label}")

        if stats["sources"]:
            with st.expander("📄 Indexed Documents"):
                for src in stats["sources"]:
                    st.markdown(f"• {src}")

        if stats["total_chunks"] > 0:
            if st.button("🗑️ Clear Knowledge Base", use_container_width=True, key="btn_clear_kb"):
                if st.session_state.get("confirm_clear"):
                    clear_index()
                    st.session_state.uploaded_files_processed = []
                    st.session_state.confirm_clear = False
                    st.success("Knowledge base cleared.")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm deletion.")

        st.divider()

        # ── User profile snapshot ────────────────────────────────────────────
        st.markdown('<div class="sidebar-header">👤 User Profile</div>', unsafe_allow_html=True)
        profile: UserProfile = st.session_state.profile
        if profile.profile_complete:
            st.markdown(
                f"""
                <div class="success-badge">
                ✅ <b>{profile.name}</b><br/>
                🍽️ {profile.food_preference}<br/>
                📍 {profile.location or '—'}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("🔧 Complete your profile for personalised recipes.")

        st.divider()
        st.markdown(
            f'<div class="footer">v{APP_VERSION} · Powered by IBM Granite</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: HOME
# ═════════════════════════════════════════════════════════════════════════════

def render_home() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <div style="font-size:3.5rem;">🍽️</div>
            <h1 style="font-size:2.2rem; margin:0.3rem 0;">AI Recipe Generator</h1>
            <p style="color:#7f8c8d; font-size:1.05rem;">
                Personalized recipes powered by <b>IBM Granite LLM</b> & Retrieval-Augmented Generation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    cols = st.columns(4)
    features = [
        ("📄", "Document RAG", "Upload recipe PDFs, DOCX & TXT files to build a personalised knowledge base."),
        ("🧑‍🍳", "Recipe Generation", "Get complete recipes with ingredients, steps, substitutions & tips."),
        ("🥗", "Nutrition Analysis", "Detailed nutritional breakdowns aligned with your health goals."),
        ("💬", "Chat Q&A", "Ask any cooking question — answered using your uploaded documents."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f"""
            <div class="recipe-card" style="text-align:center; height:160px;">
                <div style="font-size:2rem;">{icon}</div>
                <b style="color:#2c3e50;">{title}</b>
                <p style="font-size:0.85rem; color:#6c757d; margin-top:0.4rem;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick-start checklist
    st.subheader("🚀 Quick Start")
    profile: UserProfile = st.session_state.profile
    stats = get_index_stats()

    steps = [
        ("✅" if profile.profile_complete else "⬜", "Complete your **User Profile** (Step 1 tab)"),
        ("✅" if stats["document_count"] > 0 else "⬜", "Upload **Recipe Documents** (Step 2 tab) — optional but recommended"),
        ("✅" if profile.profile_complete else "⬜", "Generate your first **Personalised Recipe** (Step 3 tab)"),
        ("⬜", "Explore **Nutrition Analysis**, **Recipe Adaptation**, and **Q&A Chat**"),
    ]
    for icon, text in steps:
        st.markdown(f"{icon} {text}")

    st.divider()
    st.markdown(
        """
        <div class="disclaimer-box">
        ⚕️ <b>Important Notices:</b><br/>
        This application is for informational and culinary inspiration purposes only.
        All nutritional information is approximate. Always consult a qualified healthcare
        professional for medical dietary advice. Verify allergen information on all
        product labels before consumption.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: USER PROFILE
# ═════════════════════════════════════════════════════════════════════════════

def render_user_profile() -> None:
    st.header("👤 Your Profile")
    st.caption("Tell us about yourself so we can generate perfectly personalised recipes.")

    profile: UserProfile = st.session_state.profile

    with st.form("profile_form"):
        st.subheader("📋 Basic Information")
        col1, col2, col3 = st.columns(3)
        name = col1.text_input("Full Name *", value=profile.name, placeholder="e.g. Alex Johnson")
        age = col2.number_input("Age *", min_value=1, max_value=120, value=max(profile.age, 1), step=1)
        location = col3.text_input("Location / Country", value=profile.location, placeholder="e.g. Sydney, Australia")

        st.subheader("🍽️ Dietary Preferences")
        col4, col5 = st.columns(2)
        food_pref = col4.selectbox(
            "Food Preference *",
            FOOD_PREFERENCE_OPTIONS,
            index=FOOD_PREFERENCE_OPTIONS.index(profile.food_preference)
            if profile.food_preference in FOOD_PREFERENCE_OPTIONS
            else 0,
        )
        skill = col5.selectbox(
            "Cooking Skill Level",
            SKILL_LEVEL_OPTIONS,
            index=SKILL_LEVEL_OPTIONS.index(profile.cooking_skill_level)
            if profile.cooking_skill_level in SKILL_LEVEL_OPTIONS
            else 1,
        )

        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            DIETARY_RESTRICTION_OPTIONS,
            default=profile.dietary_restrictions,
            help="Select all that apply.",
        )

        allergies = st.multiselect(
            "⚠️ Food Allergies",
            COMMON_ALLERGEN_OPTIONS,
            default=profile.allergies,
            help="These will be flagged in all generated recipes.",
        )

        st.subheader("🌍 Culinary Preferences")
        cuisine_pref = st.multiselect(
            "Preferred Cuisines",
            CUISINE_OPTIONS,
            default=profile.cuisine_preference,
        )
        cooking_time = st.selectbox(
            "Preferred Cooking Time",
            COOKING_TIME_OPTIONS,
            index=COOKING_TIME_OPTIONS.index(profile.cooking_time_preference)
            if profile.cooking_time_preference in COOKING_TIME_OPTIONS
            else 3,
        )

        st.subheader("🥦 Ingredients & Health Goals")
        available_ingredients = st.text_area(
            "Available Ingredients",
            value=profile.available_ingredients,
            placeholder="e.g. chicken breast, garlic, tomatoes, pasta, olive oil, onions…",
            help="List ingredients you have on hand — we'll use them in your recipes.",
            height=100,
        )
        health_goals = st.multiselect(
            "Health Goals",
            HEALTH_GOAL_OPTIONS,
            default=profile.health_goals,
        )
        calorie_target_enabled = st.checkbox(
            "Set daily calorie target per meal",
            value=profile.calorie_target is not None,
        )
        calorie_target = None
        if calorie_target_enabled:
            calorie_target = st.slider(
                "Calories per meal (kcal)",
                min_value=200,
                max_value=1500,
                value=profile.calorie_target or 500,
                step=50,
            )

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary")

    if submitted:
        if not name.strip():
            st.error("Please enter your name to save the profile.")
        else:
            st.session_state.profile = UserProfile(
                name=name.strip(),
                age=int(age),
                location=location.strip(),
                food_preference=food_pref,
                dietary_restrictions=dietary_restrictions,
                allergies=allergies,
                cuisine_preference=cuisine_pref,
                available_ingredients=available_ingredients.strip(),
                cooking_skill_level=skill,
                cooking_time_preference=cooking_time,
                health_goals=health_goals,
                calorie_target=calorie_target,
                profile_complete=True,
            )
            st.success(f"✅ Profile saved for **{name.strip()}**! You're ready to generate personalised recipes.")

    # Preview
    if profile.profile_complete:
        st.divider()
        st.subheader("📄 Profile Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Age:** {profile.age}")
            st.markdown(f"**Location:** {profile.location or '—'}")
            st.markdown(f"**Food Preference:** {profile.food_preference}")
            st.markdown(f"**Skill Level:** {profile.cooking_skill_level}")
        with col_b:
            st.markdown(f"**Allergies:** {', '.join(profile.allergies) or 'None'}")
            st.markdown(f"**Dietary Restrictions:** {', '.join(profile.dietary_restrictions) or 'None'}")
            st.markdown(f"**Cuisines:** {', '.join(profile.cuisine_preference) or 'Any'}")
            st.markdown(f"**Health Goals:** {', '.join(profile.health_goals) or '—'}")

    st.divider()
    st.markdown(
        '<div class="disclaimer-box">' + ALLERGY_DISCLAIMER + "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: DOCUMENT UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

def render_document_upload() -> None:
    st.header("📚 Knowledge Base — Upload Recipe Documents")
    st.caption(
        "Upload your own recipe books, meal plans, or culinary documents. "
        "The AI will use these as a personalised knowledge source."
    )

    # Current index stats
    stats = get_index_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Documents Indexed", stats["document_count"])
    col2.metric("🔢 Text Chunks", stats["total_chunks"])
    col3.metric("📝 Supported Types", "PDF · DOCX · TXT")

    st.divider()
    st.subheader("⬆️ Upload New Documents")

    uploaded_files = st.file_uploader(
        "Drop your recipe files here",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT. Multiple files allowed.",
        key="file_uploader",
    )

    if uploaded_files:
        process_btn = st.button(
            f"🔄 Process {len(uploaded_files)} File(s) → Add to Knowledge Base",
            type="primary",
            use_container_width=True,
        )

        if process_btn:
            processed_count = 0
            skipped_count = 0
            indexed_sources = list_indexed_sources()

            progress = st.progress(0, text="Processing files…")
            status_area = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processing: {uploaded_file.name}",
                )

                if uploaded_file.name in indexed_sources:
                    status_area.info(f"⏭️ Skipping **{uploaded_file.name}** — already indexed.")
                    skipped_count += 1
                    time.sleep(0.3)
                    continue

                try:
                    file_bytes = uploaded_file.read()
                    result = process_uploaded_file(uploaded_file.name, file_bytes)

                    chunks_added = add_documents_to_index(
                        result["chunks"],
                        result["metadata"],
                        result["filename"],
                    )

                    st.session_state.uploaded_files_processed.append(uploaded_file.name)
                    status_area.success(
                        f"✅ **{uploaded_file.name}** — "
                        f"{len(result['chunks'])} chunks extracted, {chunks_added} added to index."
                    )
                    processed_count += 1

                    with st.expander(f"📄 Preview: {uploaded_file.name}"):
                        st.text(get_document_summary(result["raw_text"]))

                except Exception as exc:
                    status_area.error(f"❌ Error processing **{uploaded_file.name}**: {exc}")

                time.sleep(0.2)

            progress.empty()
            if processed_count > 0 or skipped_count > 0:
                st.success(
                    f"✅ Done! {processed_count} new file(s) indexed, {skipped_count} skipped."
                )
                st.rerun()

    # Show indexed document list
    if stats["sources"]:
        st.divider()
        st.subheader("📋 Indexed Documents")
        for src in stats["sources"]:
            st.markdown(f"📄 {src}")

    st.divider()
    st.markdown(
        """
        <div class="disclaimer-box">
        📌 <b>Privacy Note:</b> Uploaded documents are processed locally and stored in an 
        on-device vector database. Document content is used solely to answer your cooking questions 
        and generate personalised recipes. No content is shared externally beyond IBM watsonx.ai 
        API calls for text generation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: RECIPE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def render_recipe_generator() -> None:
    st.header("🧑‍🍳 Recipe Generator")
    st.caption("Describe what you want to cook and get a complete, personalised recipe.")

    profile: UserProfile = st.session_state.profile

    if not profile.profile_complete:
        st.warning("🔔 Complete your **User Profile** (Step 1) for personalised recipes.")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        recipe_query = st.text_area(
            "What would you like to cook? 🍳",
            placeholder=(
                "e.g. 'A quick pasta dish using chicken and tomatoes' or "
                "'Give me a vegan dessert for a dinner party' or "
                "'Suggest a high-protein breakfast'"
            ),
            height=110,
            key="recipe_query_input",
        )

    with col_right:
        st.markdown("**⚙️ Options**")
        use_docs = st.checkbox(
            "Use uploaded documents",
            value=True,
            help="Use your recipe knowledge base for context.",
        )
        top_k = st.slider("Context chunks", 1, 8, 5, help="Number of document chunks to retrieve.")
        recipe_language = st.selectbox(
            "🌐 Output Language",
            SUPPORTED_LANGUAGES,
            index=0,
            key="recipe_language",
            help="Generate the recipe in your preferred language.",
        )

    col_gen, col_shop = st.columns([2, 1])

    with col_gen:
        generate_btn = st.button(
            "🍽️ Generate Recipe",
            type="primary",
            use_container_width=True,
            disabled=not recipe_query.strip(),
        )

    with col_shop:
        shopping_btn = st.button(
            "🛒 Shopping List",
            use_container_width=True,
            disabled=not st.session_state.last_recipe,
        )

    # Generate recipe
    if generate_btn and recipe_query.strip():
        with st.spinner("👨‍🍳 Generating your personalised recipe…"):
            result = recipe_agent.generate(
                query=recipe_query,
                profile=profile,
                use_documents=use_docs,
                top_k=top_k,
                language=recipe_language,
            )

        st.session_state.last_recipe = result["recipe"]

        # Allergy warning (prominent)
        if result["allergy_warnings"]:
            st.markdown(
                f'<div class="allergy-alert">🚨 <b>Allergen Alert:</b> '
                f'This recipe may contain: <b>{", ".join(result["allergy_warnings"])}</b>. '
                f'Please review ingredients carefully.</div>',
                unsafe_allow_html=True,
            )

        # Recipe output
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        st.markdown(result["recipe"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Disclaimer
        if result["disclaimer"]:
            st.markdown(
                f'<div class="disclaimer-box">{result["disclaimer"]}</div>',
                unsafe_allow_html=True,
            )

        # Retrieved context expander
        if result["retrieved_chunks"]:
            with st.expander(f"📚 Retrieved Context ({len(result['retrieved_chunks'])} chunks)"):
                for chunk in result["retrieved_chunks"]:
                    st.markdown(
                        f"**Source:** {chunk['source']} | **Score:** {chunk['score']:.3f}"
                    )
                    st.text(chunk["text"][:300] + "…")
                    st.divider()

    # Show existing recipe if available
    elif st.session_state.last_recipe and not generate_btn:
        st.subheader("📋 Last Generated Recipe")
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_recipe)
        st.markdown("</div>", unsafe_allow_html=True)

    # Shopping list generation
    if shopping_btn and st.session_state.last_recipe:
        with st.spinner("🛒 Generating shopping list…"):
            shopping_list = recipe_agent.generate_shopping_list(
                st.session_state.last_recipe, profile
            )
        st.subheader("🛒 Shopping List")
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        st.markdown(shopping_list)
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB: NUTRITION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def render_nutrition_analysis() -> None:
    st.header("🥗 Nutrition Analysis")
    st.caption("Get detailed nutritional information and health insights for any recipe.")

    profile: UserProfile = st.session_state.profile

    tab_analyse, tab_mealplan = st.tabs(["🔍 Analyse a Recipe", "📅 Meal Plan Generator"])

    with tab_analyse:
        st.subheader("Paste or use a generated recipe for analysis")

        recipe_input = st.text_area(
            "Recipe Text",
            value=st.session_state.last_recipe,
            height=200,
            placeholder="Paste a recipe here or generate one from the Recipe Generator tab.",
            key="nutrition_recipe_input",
        )

        nutrition_language = st.selectbox(
            "🌐 Output Language",
            SUPPORTED_LANGUAGES,
            index=0,
            key="nutrition_language",
            help="Get the nutrition analysis in your preferred language.",
        )

        analyse_btn = st.button(
            "🔬 Analyse Nutrition",
            type="primary",
            use_container_width=True,
            disabled=not recipe_input.strip(),
        )

        if analyse_btn and recipe_input.strip():
            with st.spinner("🔬 Analysing nutritional content…"):
                analysis = nutrition_agent.analyse(recipe_input, profile, language=nutrition_language)
            st.session_state.last_nutrition = analysis

            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(analysis)
            st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.last_nutrition:
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(st.session_state.last_nutrition)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )

    with tab_mealplan:
        st.subheader("Generate a Personalised Meal Plan")

        if not profile.profile_complete:
            st.warning("Please complete your user profile first for a personalised meal plan.")

        days = st.slider("Number of days", 1, 7, 3)
        mealplan_btn = st.button(
            "📅 Generate Meal Plan",
            type="primary",
            use_container_width=True,
            disabled=not profile.profile_complete,
        )

        if mealplan_btn:
            with st.spinner(f"📅 Creating your {days}-day meal plan…"):
                meal_plan = nutrition_agent.get_meal_plan_suggestion(profile, days)
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(meal_plan)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: RECIPE ADAPTATION / PERSONALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def render_personalization() -> None:
    st.header("✨ Recipe Adaptation & Personalization")
    st.caption("Adapt any recipe to your dietary needs, swap ingredients, or reimagine it in a new cuisine.")

    profile: UserProfile = st.session_state.profile

    tab_adapt, tab_sub, tab_cuisine = st.tabs([
        "🔄 Adapt Recipe",
        "🔁 Ingredient Substitutions",
        "🌍 Cuisine Remix",
    ])

    # ── Adapt Recipe ──────────────────────────────────────────────────────────
    with tab_adapt:
        recipe_to_adapt = st.text_area(
            "Recipe to Adapt",
            value=st.session_state.last_recipe,
            height=180,
            placeholder="Paste a recipe or use the one generated in the Recipe Generator tab.",
            key="adapt_recipe_input",
        )
        adaptation_request = st.text_input(
            "Additional Adaptation Request (optional)",
            placeholder="e.g. 'Make it vegan', 'Reduce calories by 30%', 'Halve the sugar'",
        )
        adapt_btn = st.button(
            "✨ Adapt Recipe",
            type="primary",
            use_container_width=True,
            disabled=not recipe_to_adapt.strip(),
        )

        if adapt_btn and recipe_to_adapt.strip():
            with st.spinner("✨ Personalising recipe…"):
                adapted = personalization_agent.adapt_recipe(
                    recipe_to_adapt, profile, adaptation_request
                )
            st.session_state.last_adapted_recipe = adapted
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(adapted)
            st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.last_adapted_recipe:
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(st.session_state.last_adapted_recipe)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Ingredient Substitutions ─────────────────────────────────────────────
    with tab_sub:
        st.subheader("Find Substitutes for an Ingredient")
        ingredient_input = st.text_input(
            "Ingredient",
            placeholder="e.g. butter, eggs, all-purpose flour, heavy cream…",
            key="sub_ingredient_input",
        )
        sub_btn = st.button(
            "🔁 Find Substitutions",
            type="primary",
            use_container_width=True,
            disabled=not ingredient_input.strip(),
        )
        if sub_btn and ingredient_input.strip():
            with st.spinner("🔍 Finding substitutions…"):
                subs = personalization_agent.suggest_substitutions(ingredient_input, profile)
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(subs)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Cuisine Remix ────────────────────────────────────────────────────────
    with tab_cuisine:
        st.subheader("Reimagine a Recipe in a New Cuisine Style")
        cuisine_recipe = st.text_area(
            "Original Recipe",
            value=st.session_state.last_recipe,
            height=150,
            placeholder="Paste a recipe to reimagine.",
            key="cuisine_recipe_input",
        )
        target_cuisine = st.selectbox("Target Cuisine", CUISINE_OPTIONS)
        cuisine_btn = st.button(
            "🌍 Reimagine Recipe",
            type="primary",
            use_container_width=True,
            disabled=not cuisine_recipe.strip(),
        )
        if cuisine_btn and cuisine_recipe.strip():
            with st.spinner(f"🌍 Reimagining as {target_cuisine} cuisine…"):
                remixed = personalization_agent.adapt_for_cuisine(
                    cuisine_recipe, target_cuisine, profile
                )
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown(remixed)
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        f'<div class="disclaimer-box">{ALLERGY_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: CONVERSATIONAL Q&A CHAT
# ═════════════════════════════════════════════════════════════════════════════

def render_chat() -> None:
    st.header("💬 Cooking Q&A Chat")
    st.caption(
        "Ask any cooking question. The AI uses your uploaded recipe documents "
        "AND IBM Granite's culinary knowledge to answer."
    )

    profile: UserProfile = st.session_state.profile

    # Chat display
    chat_container = st.container(height=460)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align:center; color:#adb5bd; padding:3rem 1rem;">
                    <div style="font-size:2.5rem;">💬</div>
                    <p>Start a conversation! Ask me anything about cooking, recipes, 
                    ingredients, techniques, or nutritional information.</p>
                    <p style="font-size:0.85rem;">
                    <i>Examples: "How do I make pasta from scratch?" · "What's a good 
                    substitute for eggs in baking?" · "How long should I marinate chicken?"</i>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])

    # Input area
    col_input, col_clear = st.columns([5, 1])
    with col_input:
        user_message = st.chat_input("Ask a cooking question…", key="chat_input")
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Options row
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    use_docs_chat = col_opt1.checkbox("Use knowledge base", value=True, key="chat_use_docs")
    technique_mode = col_opt2.checkbox("Explain a technique", value=False, key="chat_technique")

    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        with st.spinner("🤔 Thinking…"):
            if technique_mode:
                answer_text = qa_agent.explain_technique(user_message)
                retrieved = []
            else:
                qa_result = qa_agent.answer(
                    question=user_message,
                    profile=profile if profile.profile_complete else None,
                    conversation_history=st.session_state.chat_history[:-1],
                    use_documents=use_docs_chat,
                )
                answer_text = qa_result["answer"]
                retrieved = qa_result["retrieved_chunks"]

        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})

        # Show retrieved sources if any
        if retrieved:
            sources_note = (
                "\n\n---\n📚 *Sources: "
                + ", ".join({c["source"] for c in retrieved})
                + "*"
            )
            st.session_state.chat_history[-1]["content"] += sources_note

        st.rerun()

    # Suggested questions
    st.divider()
    st.subheader("💡 Suggested Questions")
    suggested = [
        "What are common baking substitutes for eggs?",
        "How do I properly caramelise onions?",
        "What temperature should chicken be cooked to?",
        "What spices are essential in Indian cooking?",
        "How do I make a dish less spicy?",
        "What's the difference between braising and stewing?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggested):
        if cols[i % 3].button(suggestion, key=f"suggest_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            with st.spinner("🤔 Thinking…"):
                qa_result = qa_agent.answer(
                    question=suggestion,
                    profile=profile if profile.profile_complete else None,
                    conversation_history=st.session_state.chat_history[:-1],
                    use_documents=True,
                )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": qa_result["answer"]}
            )
            st.rerun()

    st.divider()
    st.markdown(
        f'<div class="disclaimer-box">{FOOD_SAFETY_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB: ABOUT / DISCLAIMERS
# ═════════════════════════════════════════════════════════════════════════════

def render_about() -> None:
    st.header("ℹ️ About This Application")

    st.markdown(
        """
        ## 🍽️ AI Recipe Generator — Document Q&A RAG Agent

        This application is a **Retrieval-Augmented Generation (RAG)** powered culinary AI
        built with **IBM Granite LLM** via **IBM watsonx.ai**.

        ---

        ### 🏗️ Architecture
        """,
        unsafe_allow_html=False,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Technology Stack**
            - 🤖 **LLM:** IBM Granite 3 8B Instruct
            - ☁️ **Platform:** IBM watsonx.ai (au-syd)
            - 🖼️ **UI Framework:** Streamlit
            - 🗄️ **Vector Store:** ChromaDB (local)
            - 🔡 **Embeddings:** sentence-transformers (MiniLM-L6-v2)
            - 📄 **Doc Parsing:** PyPDF2, python-docx
            """
        )
    with col2:
        st.markdown(
            """
            **Agentic AI Modules**
            - 🧑‍🍳 Recipe Generation Agent
            - 🥗 Nutrition Analysis Agent
            - ✨ Personalization & Adaptation Agent
            - 💬 Conversational Q&A Agent
            - 🛡️ Safety Guardrail Layer
            """
        )

    st.markdown(
        """
        ---
        ### 🔄 RAG Pipeline

        1. **Document Ingestion** — User uploads PDF/DOCX/TXT recipe documents
        2. **Chunking** — Documents split into overlapping text chunks (800 chars, 150 overlap)
        3. **Embedding** — Each chunk encoded to dense vectors (384-dim MiniLM)
        4. **Indexing** — Vectors stored in ChromaDB with cosine similarity metric
        5. **Retrieval** — On query, top-K most relevant chunks retrieved
        6. **Augmentation** — Retrieved context + user profile injected into LLM prompt
        7. **Generation** — IBM Granite generates personalised response

        ---
        ### 🤝 AI Ethics & Safety Guardrails

        This application follows **Helpful, Accurate, and Mindful (HAM)** AI principles:

        | Principle | Implementation |
        |-----------|---------------|
        | **Helpful** | Detailed recipes, explanations, and proactive suggestions |
        | **Accurate** | RAG grounds responses in real documents; medical disclaimers provided |
        | **Mindful** | Allergy detection, content filters, respectful and inclusive tone |
        | **Safe** | Keyword-based safety filter blocks harmful content |
        | **Transparent** | Retrieved document sources shown; AI nature disclosed |
        | **Private** | Documents processed locally; no user data retained externally |

        ---
        ### ⚠️ Disclaimers
        """
    )

    st.markdown(
        f'<div class="disclaimer-box">{FOOD_SAFETY_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="disclaimer-box">{ALLERGY_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ---
        ### 📜 Terms of Use

        - This tool is intended for informational and culinary inspiration purposes only.
        - Always consult a qualified healthcare professional for medical dietary advice.
        - The developers are not responsible for any adverse reactions due to incorrect allergy
          information or food preparation errors.
        - IBM Granite LLM responses are AI-generated and should be verified before acting on them.

        ---
        """,
        unsafe_allow_html=False,
    )

    st.markdown(
        f"""
        <div style="text-align:center; color:#7f8c8d; font-size:0.85rem; padding-top:1rem;">
            <span class="ibm-badge">🤖 IBM Granite LLM</span>&nbsp;
            <b>{APP_NAME}</b> · v{APP_VERSION}<br/>
            Built with ❤️ using IBM watsonx.ai &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Tab navigation
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    render_sidebar()

    tab_home, tab_profile, tab_docs, tab_recipe, tab_nutrition, tab_adapt, tab_chat, tab_about = st.tabs([
        "🏠 Home",
        "👤 Step 1: Profile",
        "📚 Step 2: Documents",
        "🧑‍🍳 Step 3: Generate",
        "🥗 Nutrition",
        "✨ Adapt",
        "💬 Chat",
        "ℹ️ About",
    ])

    with tab_home:
        render_home()
    with tab_profile:
        render_user_profile()
    with tab_docs:
        render_document_upload()
    with tab_recipe:
        render_recipe_generator()
    with tab_nutrition:
        render_nutrition_analysis()
    with tab_adapt:
        render_personalization()
    with tab_chat:
        render_chat()
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
