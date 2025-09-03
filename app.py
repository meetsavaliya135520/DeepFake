import os
import io
import time
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ----------------------------
# Defaults (adjust in the sidebar)
# ----------------------------
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ["FAKE", "REAL"]  # index 0 -> FAKE, index 1 -> REAL


# ----------------------------
# Utilities
# ----------------------------
def is_image_file(path: str) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return os.path.splitext(path.lower())[1] in exts


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str, custom_objects: dict | None = None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    # compile=False -> faster load; not needed for inference
    return load_model(model_path, compile=False, custom_objects=custom_objects or {})


def prepare_image(img: Image.Image, target_size=(224, 224), preprocessing="resnet50") -> np.ndarray:
    """
    Convert PIL image to model-ready batch (1, H, W, 3).
    """
    img = img.convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if preprocessing == "resnet50":
        arr = preprocess_input(arr)  # BGR + mean subtraction
    elif preprocessing == "rescale_1_255":
        arr = arr / 255.0
    else:
        # raw passthrough
        pass
    return np.expand_dims(arr, axis=0)


def predict_batch(model, images: list[np.ndarray], threshold: float, class_names: list[str]) -> list[dict]:
    """
    Supports sigmoid (1 node) or softmax (2 nodes).
    Returns list of dicts with label and probabilities.
    """
    if not images:
        return []

    X = np.vstack(images)  # (N, H, W, 3)
    preds = model.predict(X, verbose=0)

    results = []
    if preds.ndim == 2 and preds.shape[1] == 1:
        # sigmoid: prob of class 1 (assume class 1 = REAL)
        p_real = preds[:, 0].astype(float)
        p_fake = 1.0 - p_real
        labels = np.where(p_real >= threshold, class_names[1], class_names[0])
        for i in range(len(X)):
            results.append({"label": str(labels[i]), "p_real": float(p_real[i]), "p_fake": float(p_fake[i])})
    elif preds.ndim == 2 and preds.shape[1] == 2:
        # softmax: index 0 -> FAKE, index 1 -> REAL
        p_fake = preds[:, 0].astype(float)
        p_real = preds[:, 1].astype(float)
        labels = np.array(class_names)[np.argmax(preds, axis=1)]
        for i in range(len(X)):
            results.append({"label": str(labels[i]), "p_real": float(p_real[i]), "p_fake": float(p_fake[i])})
    else:
        raise ValueError(f"Unexpected model output shape: {preds.shape}")
    return results


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Fake vs Real (ResNet50)", layout="centered")
st.title("🕵️‍♂️ Fake vs Real — ResNet50 ")

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model", "resnet50.keras")
    img_w = st.number_input("Input width", min_value=64, max_value=1024, value=DEFAULT_IMAGE_SIZE[0], step=8)
    img_h = st.number_input("Input height", min_value=64, max_value=1024, value=DEFAULT_IMAGE_SIZE[1], step=8)
    preprocessing = st.selectbox(
        "Preprocessing",
        ["resnet50", "rescale_1_255", "none"],
        help="Choose the preprocessing that matches your training."
    )
    threshold = st.slider("Decision threshold (sigmoid models)", 0.0, 1.0, 0.5, 0.01)
    class0 = st.text_input("Class 0 name", DEFAULT_CLASS_NAMES[0])
    class1 = st.text_input("Class 1 name", DEFAULT_CLASS_NAMES[1])
    class_names = [class0.strip(), class1.strip()]
    st.caption(
        "Assumed mapping:\n- index 0 → Class 0\n- index 1 → Class 1\nFor sigmoid, threshold decides between them.")

    st.divider()
    st.subheader("Batch folder (optional)")
    folder_path = st.text_input("Folder path (predict all images inside)", "")

col_left, col_right = st.columns([1, 1])

# Load model
model = None
load_err = None
if model_path:
    try:
        with st.spinner("Loading model..."):
            model = load_keras_model(model_path)
        st.success("Model loaded.")
    except Exception as e:
        load_err = str(e)
        st.error(load_err)

# Single image upload
with col_left:
    st.subheader("Single Image")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    if model and file is not None:
        try:
            pil_img = Image.open(io.BytesIO(file.read()))
            st.image(pil_img, caption="Input", use_container_width=True)
            x = prepare_image(pil_img, (int(img_w), int(img_h)), preprocessing)
            t0 = time.time()
            result = predict_batch(model, [x], threshold, class_names)[0]
            dt = (time.time() - t0) * 1000
            st.markdown(
                f"**Prediction:** `{result['label']}`  •  P(REAL)={result['p_real']:.4f}  |  P(FAKE)={result['p_fake']:.4f}  •  _{dt:.1f} ms_")
        except Exception as e:
            st.error(f"Failed to predict: {e}")

# Batch folder
with col_right:
    st.subheader("Batch Folder")
    if folder_path:
        if not os.path.isdir(folder_path):
            st.warning("Folder not found.")
        elif model:
            paths = []
            tensors = []
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if is_image_file(fpath):
                        try:
                            img = Image.open(fpath)
                            tensors.append(prepare_image(img, (int(img_w), int(img_h)), preprocessing))
                            paths.append(fpath)
                        except Exception as e:
                            st.warning(f"Skip {fname}: {e}")

            if tensors:
                with st.spinner(f"Predicting {len(tensors)} image(s)..."):
                    results = predict_batch(model, tensors, threshold, class_names)
                # Table
                import pandas as pd

                df = pd.DataFrame({
                    "path": paths,
                    "label": [r["label"] for r in results],
                    "P(REAL)": [r["p_real"] for r in results],
                    "P(FAKE)": [r["p_fake"] for r in results],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
                # Simple summary
                counts = df["label"].value_counts().to_dict()
                st.write("**Summary:**", counts)
            else:
                st.info("No valid images found in that folder.")

