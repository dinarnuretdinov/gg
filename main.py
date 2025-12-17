import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Распознавание изображений с помощью Hugging Face")

url = "https://t.me/+qHHDOyGAj9llM2Qy"
st.write("Ссылка для перехода в чат с материалами [тык](%s)" % url)

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png"]
)

# Кэшируем модель
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    classifier = load_model()

    with st.spinner("Распознаю изображение..."):
        results = classifier(image)

    st.write("### Результаты распознавания")
    for result in results:
        st.write(f"**{result['label']}**: {result['score']:.2f}")
