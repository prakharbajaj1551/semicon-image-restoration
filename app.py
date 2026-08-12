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

from src.dataset.degradation import DegradationSettings, degrade_image
from inference import load_model, restore_image
from src.utils.image_io import npy_to_tensor, pil_to_tensor
from src.utils.metrics import psnr

# The organisers' data comes as .npy arrays, so the app accepts those
# too — judges can drop an actual test file straight into the demo.
UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"]


def load_uploaded(uploaded_file):
    """Uploaded file -> (1, H, W) tensor, whatever the format was."""
    if uploaded_file.name.lower().endswith(".npy"):
        return npy_to_tensor(np.load(uploaded_file))
    return pil_to_tensor(Image.open(uploaded_file))

CHECKPOINT = "weights/model.pth"

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


def to_display(tensor, match_size=None):
    """(1,H,W) tensor in [0,1] -> 2D array Streamlit can show as an image.

    WHY match_size matters (this is not cosmetic):
        Streamlit stretches every image to the column width. A 128x128
        degraded image stretched to ~600 px gets SMOOTHED by the browser,
        which hides its noise — the input then looks almost as clean as
        the restoration and the model appears to do nothing.
        Passing match_size repeats each pixel (nearest neighbour) so the
        small image keeps its real, blocky, noisy pixels. Now the
        comparison shows what the model actually removed.
    """
    array = tensor.squeeze(0).clamp(0, 1).numpy()
    if match_size is not None and array.shape[0] != match_size:
        factor = match_size // array.shape[0]
        array = np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)
    return array


def zoom_crop(tensor, match_size=None, fraction=4):
    """Center crop covering 1/`fraction` of the image, for a detail view.

    WHY: on a full 256x256 view shown small on screen, fine differences are
    hard to see. Zooming into the middle makes noise and edge sharpness
    obvious — this is what convinces a judge.
    """
    array = to_display(tensor, match_size)
    h, w = array.shape
    ch, cw = h // fraction, w // fraction
    top, left = (h - ch) // 2, (w - cw) // 2
    return array[top:top + ch, left:left + cw]


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

    # The input/ground-truth pair can come from an upload OR from the
    # built-in sample picker below — both feed the same display code.
    degraded = truth = None
    if uploaded is not None:
        degraded = load_uploaded(uploaded)
        if reference is not None:
            truth = load_uploaded(reference)
    elif Path("data/test/lq").exists():
        # One-click demo: judges shouldn't watch us dig through folders.
        # Lists held-out test files; the matching ground truth is loaded
        # automatically so PSNR appears without a second upload.
        from src.utils.image_io import list_images, load_image
        sample_names = [p.name for p in list_images("data/test/lq")[:20]]
        choice = st.selectbox("…or pick a sample from the held-out test set",
                              ["(choose a sample)"] + sample_names)
        if choice != "(choose a sample)":
            degraded = load_image(Path("data/test/lq") / choice)
            gt_path = Path("data/test/gt") / choice
            if gt_path.exists():
                truth = load_image(gt_path)

    if degraded is not None:
        start = time.time()
        restored = restore_image(model, degraded, scale, device,
                                 tile=256, overlap=32)
        seconds = time.time() - start

        # The bicubic baseline: what you get with NO AI, for a fair 3-way view.
        baseline = F.interpolate(degraded.unsqueeze(0), scale_factor=scale,
                                 mode="bicubic", align_corners=False)
        baseline = baseline.squeeze(0).clamp(0, 1)

        big = restored.shape[-1]   # display every panel at this pixel size
        left, middle, right = st.columns(3)
        with left:
            st.subheader("Degraded input")
            st.image(to_display(degraded, big), use_container_width=True)
            st.caption(f"{degraded.shape[-1]} x {degraded.shape[-2]} px "
                       f"(shown {scale}x, real pixels)")
        with middle:
            st.subheader(f"Bicubic x{scale} (no AI)")
            st.image(to_display(baseline), use_container_width=True)
            st.caption("plain upscaling — noise is still there")
        with right:
            st.subheader("Restored (our model)")
            st.image(to_display(restored), use_container_width=True)
            st.caption(f"{restored.shape[-1]} x {restored.shape[-2]} px")

        if st.checkbox("Zoom into the centre detail", value=True):
            zoom_cols = st.columns(3)
            zoom_cols[0].image(zoom_crop(degraded, big), use_container_width=True,
                               caption="degraded (zoom)")
            zoom_cols[1].image(zoom_crop(baseline), use_container_width=True,
                               caption="bicubic (zoom)")
            zoom_cols[2].image(zoom_crop(restored), use_container_width=True,
                               caption="ours (zoom)")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Inference time", f"{seconds:.2f} s")
        metric_cols[1].metric("Upscaling", f"x{scale}")
        if truth is not None:
            if truth.shape == restored.shape:
                # Show ours AND the baseline so the gain is explicit.
                ours_psnr = psnr(restored, truth)
                base_psnr = psnr(baseline, truth)
                metric_cols[2].metric("PSNR vs ground truth",
                                      f"{ours_psnr:.2f} dB",
                                      delta=f"{ours_psnr - base_psnr:+.2f} dB "
                                            f"vs bicubic ({base_psnr:.2f})")
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
