import streamlit as st
from transformers import pipeline
from PIL import Image
#Заголовок приложения
st.title("Распознавание изображений с помощью Hugging Face")

#Добавим ссылку на чат в телеграм
url = "https://t.me/+qHHDOyGAj9llM2Qy"
st.write("Ссылка для перехода в чат с материалами [тык](%s)" % url)

#Добавляем блок с загрузкой изображения от пользователя
uploaded_file = st.file_uploader("Загрузите изображение", type = ["jpg", "jpeg", "png"])

#Проверка, загружено ли изображение
if uploaded_file is not None:
    # Открываем изображение с помощью PIL и выводим на экран
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width = True)

    # Инициализация модели для классификации изображений
    classifier = pipeline("image-classification", model="./local_vit")

    # Распознавание изображения и запись результатов
    results  = classifier(image)

    st.write("**Результаты распознавания**")
    for result in results:
        st.write(f"{result['label']}: {result['score']:.2f}")
        
