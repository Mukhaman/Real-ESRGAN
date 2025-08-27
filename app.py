import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os

from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# -------------------------------
# Load RealESRGAN model
# -------------------------------
@st.cache_resource
def load_model():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4
    )

    model_path = "weights/net_g_latest.pth"  # put your .pth file here
    if not os.path.isfile(model_path):
        st.error(f"Model weights not found at {model_path}")
        st.stop()

    use_cuda = torch.cuda.is_available()
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=100,        # try 0 first; if issues, 128–256 is safe
        tile_pad=10,
        pre_pad=0,
        half=use_cuda,   # only use fp16 if CUDA exists
        gpu_id=0 if use_cuda else None
    )
    return upsampler

upsampler = load_model()


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Sentinel-2 Super-Resolution (×4 with Real-ESRGAN)")
st.write("Upload a low-resolution Sentinel-2 RGB image to upscale by 4×")

uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","tif"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    # To model: contiguous BGR uint8
    img_bgr = np.ascontiguousarray(np.array(pil_img)[:, :, ::-1])

    st.image(pil_img, caption="Low-resolution Input", use_container_width=True)

    if st.button("Upscale"):
        with st.spinner("Upscaling... please wait ⏳"):
            try:
                output, _ = upsampler.enhance(img_bgr, outscale=4)

                # --- Robust dtype handling ---
                if output.dtype != np.uint8:
                    # RealESRGAN usually returns uint8, but just in case
                    # clamp and convert from [0,1] floats
                    output = np.clip(output, 0, 1)
                    output = (output * 255.0).round().astype(np.uint8)

                # Sanity check (helps debug “black” issue)
                st.write(
                    "OUT dtype:", str(output.dtype),
                    "min:", int(output.min()),
                    "max:", int(output.max()),
                    "shape:", tuple(output.shape)
                )

                # BGR -> RGB for display/save
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                result_img = Image.fromarray(output_rgb)

            except RuntimeError as e:
                st.error(f"Error during inference: {e}")
                st.stop()

        st.image(result_img, caption="Upscaled ×4", use_container_width=True)
        st.success("Done ✅")

        # Download button
        from io import BytesIO
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            "💾 Download Upscaled Image",
            data=buf.getvalue(),
            file_name="upscaled.png",
            mime="image/png"
        )

