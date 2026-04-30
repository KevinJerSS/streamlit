import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configuración de la página
st.set_page_config(page_title="Detector de Razas de Perros", page_icon="🐶", layout="centered")

# Cargar el modelo preentrenado
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y no cada vez que interactúas con la app
@st.cache_resource
def cargar_modelo():
    return MobileNetV2(weights='imagenet')

modelo = cargar_modelo()

st.title("🐶 Detector de Razas de Perros con ML")
st.markdown("""
Sube una foto de un perro y este modelo de Machine Learning (MobileNetV2) analizará la imagen para detectar de qué raza se trata.
*Nota: Los nombres de las razas provienen de la base de datos original y se mostrarán en inglés.*
""")

# Creador de subida de archivos
archivo_subido = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Mostrar la imagen
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Imagen a analizar', use_container_width=True)
    
    with st.spinner("Analizando la imagen..."):
        # Preprocesamiento de la imagen para que MobileNetV2 la entienda
        # 1. Convertir a RGB (por si es una imagen con fondo transparente o blanco y negro)
        # 2. Redimensionar a 224x224 píxeles
        img_procesada = imagen.convert('RGB').resize((224, 224))
        
        # Convertir a arreglo de numpy y añadir una dimensión extra (batch)
        img_array = np.array(img_procesada)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Aplicar el preprocesamiento específico del modelo
        img_array = preprocess_input(img_array)
        
        # Realizar la predicción
        predicciones = modelo.predict(img_array)
        
        # Decodificar las predicciones (obtener el top 3)
        resultados = decode_predictions(predicciones, top=3)[0]
        
        st.success("¡Análisis completado!")
        st.subheader("Resultados:")
        
        # Mostrar los resultados formateados
        for i, (imagenet_id, etiqueta, probabilidad) in enumerate(resultados):
            nombre_raza = etiqueta.replace('_', ' ').title()
            porcentaje = probabilidad * 100
            st.write(f"**{i+1}. {nombre_raza}** - Probabilidad: {porcentaje:.2f}%")
            
            # Barra de progreso visual
            st.progress(float(probabilidad))
