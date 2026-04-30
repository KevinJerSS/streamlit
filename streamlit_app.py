import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# --- Configuración de la página ---
st.set_page_config(page_title="Identificador de Razas de Perro", page_icon="🐾", layout="centered")

# --- Carga del Modelo de Machine Learning ---
# Usamos @st.cache_resource para cargar el modelo una sola vez y hacer la app muy rápida
@st.cache_resource
def cargar_modelo():
    return MobileNetV2(weights='imagenet')

modelo = cargar_modelo()

# --- Interfaz de Usuario ---
st.title("🐾 ¿Qué raza es este perro?")
st.markdown("""
Sube la foto de un perro. Nuestra Inteligencia Artificial analizará los rasgos físicos y te dirá con qué raza coincide más.
*(Los nombres de las razas se mostrarán en inglés, tal como están en la base de datos científica).*
""")

# Selector de archivos
archivo_subido = st.file_uploader("Sube una foto de un perro (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # 1. Mostrar la imagen original
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Tu imagen', use_container_width=True)
    
    with st.spinner("La IA está escaneando los rasgos de la imagen..."):
        # 2. Preprocesamiento (Preparar la foto para la IA)
        # Convertimos a RGB por si la imagen tiene canal alfa (transparente) y redimensionamos a 224x224
        img_procesada = imagen.convert('RGB').resize((224, 224))
        
        # Convertimos la imagen a una matriz de números
        img_array = np.array(img_procesada)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # 3. Predicción
        predicciones = modelo.predict(img_array)
        
        # Extraemos las 3 razas más probables
        resultados = decode_predictions(predicciones, top=3)[0]
        
        # 4. Mostrar Resultados
        st.success("¡Análisis completado!")
        st.subheader("Resultados de la IA:")
        
        for i, (id_red, raza, probabilidad) in enumerate(resultados):
            # Limpiamos el texto para que se vea bonito
            nombre_limpio = raza.replace('_', ' ').capitalize()
            porcentaje = probabilidad * 100
            
            st.write(f"**{i+1}. {nombre_limpio}**")
            st.write(f"Nivel de confianza: {porcentaje:.2f}%")
            st.progress(float(probabilidad))
