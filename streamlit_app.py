import streamlit as st
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import requests

# --- Configuración ---
st.set_page_config(page_title="Detector Ligero", page_icon="🐕", layout="centered")

st.title("🐕 Detector de Razas (Versión Ultraligera)")
st.markdown("Esta versión usa **MobileNet V3 Small** y PyTorch CPU. Está optimizada para gastar el mínimo de memoria y procesar las imágenes en milisegundos.")

# --- Cargar Modelo Eficiente ---
@st.cache_resource
def cargar_recursos():
    # 1. Cargamos los pesos del modelo más ligero disponible
    pesos = MobileNet_V3_Small_Weights.DEFAULT
    modelo = mobilenet_v3_small(weights=pesos)
    modelo.eval() # Lo ponemos en modo "solo lectura" (ahorra memoria)
    
    # 2. Obtenemos las reglas de preprocesamiento de la imagen
    preprocesar = pesos.transforms()
    
    # 3. Descargamos la lista de nombres de animales/razas de internet (ImageNet)
    url_etiquetas = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    etiquetas = requests.get(url_etiquetas).text.split('\n')
    
    return modelo, preprocesar, etiquetas

modelo, preprocesar, etiquetas = cargar_recursos()

# --- Interfaz ---
archivo = st.file_uploader("Sube una foto de un perro (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if archivo:
    imagen = Image.open(archivo).convert('RGB')
    st.image(imagen, caption="Imagen cargada", use_container_width=True)
    
    with st.spinner("Analizando a alta velocidad..."):
        # Preparar la imagen
        img_tensor = preprocesar(imagen).unsqueeze(0)
        
        # Predecir sin guardar historial matemático (ahorra muchísima RAM)
        with torch.no_grad():
            prediccion = modelo(img_tensor)
        
        # Calcular probabilidades reales (de 0 a 100%)
        probabilidades = torch.nn.functional.softmax(prediccion[0], dim=0)
        top3_prob, top3_ids = torch.topk(probabilidades, 3)
        
        st.success("¡Análisis completado!")
        st.subheader("Resultados:")
        
        # Mostrar el Top 3
        for i in range(3):
            # Limpiamos el nombre para que se lea mejor (ej. "golden_retriever" -> "Golden Retriever")
            raza = etiquetas[top3_ids[i]].replace('_', ' ').title()
            prob = top3_prob[i].item()
            
            st.write(f"**{i+1}. {raza}** - {prob * 100:.1f}%")
            st.progress(prob)
