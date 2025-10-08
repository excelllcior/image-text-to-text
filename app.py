import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Распознавание текста")

st.title("Распознавание англоязычного текста с изображения")
st.write("Загрузите изображение")

@st.cache_resource
def load_ocr_pipeline():
    try:
        pipe = pipeline(
            "image-to-text", 
            model="microsoft/trocr-base-printed",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return pipe
    except:
        pipe = pipeline(
            "image-to-text",
            model="microsoft/trocr-base-stage1",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return pipe

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    if st.button("Распознать текст"):
        try:
            with st.spinner("Распознавание текста..."):
                pipe = load_ocr_pipeline()
                result = pipe(image)
                generated_text = result[0]['generated_text']
            
            st.success("Текст распознан!")
            st.write(f"Результат: {generated_text}")
            
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.info("Попробуйте другое изображение с более четким текстом")
else:
    st.info("Загрузите изображение с текстом")