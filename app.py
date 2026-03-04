import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

st.title("Image Captioning App")

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
image = image.resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.write("### Generated Caption:")
    st.success(caption) 