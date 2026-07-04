"""
Streamlit frontend for the Siamese Visual Similarity Engine.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys, os, tempfile
from pathlib import Path

# ── Resolve project root (one level above app/) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# retrieval.py does `from model import *` so src/ must be importable
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import retrieve  # noqa: E402  (after sys.path fix)
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siamese Visual Similarity Engine",
    page_icon="🐦",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* card for each result */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.6rem;
    }
    .distance-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #7fdbca;
        padding: 4px 12px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .species-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 6px;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────
st.title("🐦 Siamese Visual Similarity Engine")
st.markdown(
    "Upload a bird photo and the system retrieves the **most visually similar** "
    "birds from the CUB-200 dataset using learned embedding distances — "
    "no class labels needed."
)

# ── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.info("💡 **Lower distance = higher visual similarity.**")

# ── Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png"],
    help="Accepts JPG or PNG bird photos.",
)

if uploaded is not None:
    # Show the query image
    st.subheader("Query image")
    st.image(uploaded, width=300)

    # Write to a temp file so retrieve() can read it by path
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    # ── Retrieve ─────────────────────────────────────────────────────────
    with st.spinner("Computing embedding & searching..."):
        try:
            results = retrieve(tmp_path, k=k)
        finally:
            os.unlink(tmp_path)  # clean up regardless

    # ── Display results ──────────────────────────────────────────────────
    st.subheader(f"Top {len(results)} similar birds")
    cols = st.columns(min(len(results), 5))

    for i, item in enumerate(results):
        col = cols[i % len(cols)]
        img_path = DATA_DIR / item["path"]
        species_raw = item["species"]
        # "062.Herring_Gull" -> "Herring Gull"
        species_display = species_raw.split(".", 1)[-1].replace("_", " ")
        distance = item["distance"]

        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Image not found:\n{img_path.name}")

            st.markdown(
                f'<p class="species-name">{species_display}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="distance-badge">d = {distance:.4f}</span>',
                unsafe_allow_html=True,
            )

    # ── Footer note ──────────────────────────────────────────────────────
    st.caption("Lower distance means higher visual similarity.")
else:
    st.markdown(
        "<br><p style='text-align:center; color:#888;'>"
        "👆 Upload a bird photo above to get started.</p>",
        unsafe_allow_html=True,
    )
