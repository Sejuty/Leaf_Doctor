"""Leaf Doctor — Streamlit UI. All model logic lives in src/leafdoctor/inference.py."""

import streamlit as st

from leafdoctor.config import LOW_CONFIDENCE_THRESHOLD, pretty_label
from leafdoctor.inference import InvalidImageError, decode_image, load_model, predict

st.set_page_config(page_title="Leaf Doctor", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Cached so the 9.8 MB model loads once per session rather than on every rerun."""
    return load_model()


st.title("🌿 Leaf Doctor")
st.write("Be a doctor for your pepper, potato or tomato plant — upload a leaf photo.")

uploaded_file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a JPG or PNG of a single leaf to get a diagnosis.")
    st.stop()

try:
    rgb = decode_image(uploaded_file.getvalue())
except InvalidImageError:
    st.error("That file could not be read as an image. Try a different JPG or PNG.")
    st.stop()

st.image(rgb, caption="Uploaded image", use_container_width=True)

try:
    results = predict(get_model(), rgb, top_k=3)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface any inference failure to the user
    st.error(f"Prediction failed: {exc}")
    st.stop()

top_label, top_prob = results[0]

if top_prob < LOW_CONFIDENCE_THRESHOLD:
    st.warning(
        f"Low confidence ({top_prob:.1%}). This may not be a pepper, potato or tomato "
        "leaf — Leaf Doctor only recognises 15 specific classes."
    )
else:
    st.success(f"**{pretty_label(top_label)}** — {top_prob:.1%} confidence")

st.subheader("Top 3 predictions")
for label, prob in results:
    st.write(pretty_label(label))
    st.progress(prob, text=f"{prob:.1%}")

st.caption(
    "For guidance only — not a substitute for professional agronomic advice."
)
