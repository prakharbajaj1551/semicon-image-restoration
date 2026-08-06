"""Streamlit demo: upload an image -> restore it -> compare + metrics.

Run with:
    streamlit run app.py

Two modes (picked in the sidebar):

  "Restore"  you upload an already-degraded image; we restore it.
             PSNR is only shown if you also upload the matching clean
             image (PSNR needs a reference to compare against).

  "Demo"     you upload a CLEAN image; the app degrades it live with our
             degradation pipeline, restores it, and shows all three side
             by side with PSNR — including the bicubic baseline, so the
             audience sees exactly what the AI adds. Use this for judging.

WHY the model is wrapped in @st.cache_resource:
    Streamlit reruns this whole script on every click. Without caching we
    would reload the network from disk each time; with it, the model
    loads once and is reused — the app stays snappy.
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from dataset.degradation import DegradationSettings, degrade_image
from inference import load_model, restore_image
from utils.image_io import npy_to_tensor, pil_to_tensor
from utils.metrics import psnr

# The organisers' data comes as .npy arrays, so the app accepts those
# too — judges can drop an actual test file straight into the demo.
UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"]


def load_uploaded(uploaded_file):
    """Uploaded file -> (1, H, W) tensor, whatever the format was."""
    if uploaded_file.name.lower().endswith(".npy"):
        return npy_to_tensor(np.load(uploaded_file))
    return pil_to_tensor(Image.open(uploaded_file))

CHECKPOINT = "checkpoints/best.pth"

# initial_sidebar_state="expanded" so the mode switch is always visible —
# with the default "auto" the sidebar can start collapsed and get missed.
st.set_page_config(page_title="Semiconductor Image Restoration",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_model():
    """Load the trained model once and cache it across reruns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(CHECKPOINT, device)
    return model, config, device


def to_display(tensor):
    """(1,H,W) tensor in [0,1] -> 2D array Streamlit can show as an image."""
    return tensor.squeeze(0).clamp(0, 1).numpy()


def to_png_bytes(tensor):
    """(1,H,W) tensor -> PNG bytes for the download button."""
    array = (tensor.squeeze(0).clamp(0, 1).numpy() * 255).round().astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- header
st.title("Semiconductor Inspection Image Restoration")
st.caption("Joint denoising (speckle + Gaussian) and super-resolution "
           "with a NAFNet-style network")

if not Path(CHECKPOINT).exists():
    st.error(f"No trained model found at `{CHECKPOINT}`. "
             "Train one first (see README) and place best.pth there.")
    st.stop()   # nothing below can work without a model

model, config, device = get_model()
scale = config["scale"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "What do you want to do?",
        ["Restore a degraded image", "Demo: degrade a clean image, then restore"],
        label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Model:** NAFNet-SR  \n"
                f"**Scale:** x{scale}  \n"
                f"**Width / blocks:** {config['width']} / {config['n_blocks']}  \n"
                f"**Device:** {device.type.upper()}")

# ---------------------------------------------------------------- mode 1
if mode == "Restore a degraded image":
    uploaded = st.file_uploader("Upload a degraded grayscale image",
                                type=UPLOAD_TYPES)
    reference = st.file_uploader(
        "Optional: upload the clean ground truth (enables PSNR)",
        type=UPLOAD_TYPES)

    if uploaded is not None:
        degraded = load_uploaded(uploaded)

        start = time.time()
        restored = restore_image(model, degraded, scale, device,
                                 tile=256, overlap=32)
        seconds = time.time() - start

        left, right = st.columns(2)
        with left:
            st.subheader("Original (degraded)")
            st.image(to_display(degraded), use_container_width=True)
            st.caption(f"{degraded.shape[-1]} x {degraded.shape[-2]} px")
        with right:
            st.subheader("Restored")
            st.image(to_display(restored), use_container_width=True)
            st.caption(f"{restored.shape[-1]} x {restored.shape[-2]} px")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Inference time", f"{seconds:.2f} s")
        metric_cols[1].metric("Upscaling", f"x{scale}")
        if reference is not None:
            truth = load_uploaded(reference)
            if truth.shape == restored.shape:
                metric_cols[2].metric("PSNR vs ground truth",
                                      f"{psnr(restored, truth):.2f} dB")
            else:
                st.warning("Ground truth size doesn't match the restored "
                           "image, so PSNR can't be computed.")

        st.download_button("Download restored image", to_png_bytes(restored),
                           file_name="restored.png", mime="image/png")

# ---------------------------------------------------------------- mode 2
else:
    st.markdown("Upload a **clean** image. We degrade it with the exact "
                "pipeline used in training (shrink + speckle + Gaussian "
                "noise), restore it back, and score both steps.")
    uploaded = st.file_uploader("Upload a clean grayscale image",
                                type=UPLOAD_TYPES)

    if uploaded is not None:
        clean = load_uploaded(uploaded)
        # Trim so the size divides evenly by the scale (same rule as training).
        clean = clean[:, : clean.shape[-2] // scale * scale,
                      : clean.shape[-1] // scale * scale]

        # Degrade with a fixed seed so re-clicks show the same corruption.
        degraded = degrade_image(clean, DegradationSettings(scale=scale), seed=7)

        start = time.time()
        restored = restore_image(model, degraded, scale, device,
                                 tile=256, overlap=32)
        seconds = time.time() - start

        # The no-AI baseline for honest comparison.
        bicubic = F.interpolate(degraded.unsqueeze(0), scale_factor=scale,
                                mode="bicubic", align_corners=False)
        bicubic = bicubic.squeeze(0).clamp(0, 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Clean original")
            st.image(to_display(clean), use_container_width=True)
            st.caption(f"{clean.shape[-1]} x {clean.shape[-2]} px")
        with col2:
            st.subheader("Degraded input")
            st.image(to_display(degraded), use_container_width=True)
            st.caption(f"{degraded.shape[-1]} x {degraded.shape[-2]} px "
                       f"(noise + {scale}x smaller)")
        with col3:
            st.subheader("AI restored")
            st.image(to_display(restored), use_container_width=True)
            st.caption(f"{restored.shape[-1]} x {restored.shape[-2]} px")

        ours = psnr(restored, clean)
        base = psnr(bicubic, clean)
        metric_cols = st.columns(3)
        metric_cols[0].metric("PSNR — bicubic baseline", f"{base:.2f} dB")
        metric_cols[1].metric("PSNR — our model", f"{ours:.2f} dB",
                              delta=f"{ours - base:+.2f} dB vs baseline")
        metric_cols[2].metric("Inference time", f"{seconds:.2f} s")

        st.download_button("Download restored image", to_png_bytes(restored),
                           file_name="restored.png", mime="image/png")
