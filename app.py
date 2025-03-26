import streamlit as st
import cv2
import numpy as np
from cv2 import dnn_superres
from PIL import Image
import io
from io import StringIO


# In[3]:
def CVDnnSuperResImpl(image,model_name,scale):
    model_paths = {
        'edsr': {
            2: './models/EDSR_x2.pb',
            3: './models/EDSR_x3.pb',
            4: './models/EDSR_x4.pb'
        },
        'lapsrn': {
            2: './models/LapSRN_x2.pb',
            4: './models/LapSRN_x4.pb',
            8: './models/LapSRN_x8.pb'
        }
    }
    
    # Initialize the super-resolution model
    model_path = model_paths[model_name][scale] 
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)
    
    # Convert PIL image to NumPy array
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    
    # Perform upscaling
    upscaled = sr.upsample(img)
    
    # Convert back to RGB for display
    upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    
    return upscaled_rgb



def upscale_image(image, model_name='edsr', scale=4):
    model_fn_dict={
         'edsr': CVDnnSuperResImpl,
        'lapsrn': CVDnnSuperResImpl
    }
    return model_fn_dict[model_name](image, model_name, scale)




    # Streamlit UI
st.title("🔍 AI Image Upscaler")

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# # Model selection
model_options = {
    "edsr": [2, 3, 4],
    "lapsrn": [2, 4, 8]
}

model_name = st.selectbox("Choose a super-resolution model", ["lapsrn"])
file_name = st.text_input("Enter a name for the upscaled image (without extension)", "upscaled_image")
# Scale selection
if model_name=="edsr":
    scale_factor = st.select_slider("Scale Factor", model_options[model_name], value=3)
elif model_name=="lapsrn":
    scale_factor = st.select_slider("Scale Factor", model_options[model_name], value=4)


if uploaded_image is not None:
    # Load the image

    image = Image.open(uploaded_image)
    
    # Display original image
    st.image(image, caption=f"Original Image Size: {image.size[1]}x{image.size[0]} Original Image", use_container_width=True)
    
    # Button to upscale image
    if st.button("Upscale Image"):
        upscaled_image = upscale_image(image, model_name, scale_factor)
       
        # Display upscaled image
     
        st.image(upscaled_image, caption=f"Upscaled Image Size: {upscaled_image.shape} Upscaled Image ({model_name.upper()} x{scale_factor})", use_container_width=True)

        # Convert upscaled image to a downloadable format
        upscaled_pil = Image.fromarray(upscaled_image)
        buf = io.BytesIO()
        upscaled_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Download button
        st.download_button(label="Download Upscaled Image", data=byte_im, file_name=f"{file_name}_{model_name}_x{scale_factor}.png", mime="image/png")
