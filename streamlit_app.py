import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Configuración de la página ---
st.set_page_config(page_title="Pipeline ML: Promociones Retail", layout="wide")

# --- Generación de Datos (Mock Data) ---
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Gasto_Abarrotes_Soles': np.random.normal(35, 15, n),
        'Frecuencia_Visitas_Mes': np.random.randint(1, 10, n),
        'Uso_Tarjeta_Fidelidad': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'Ticket_Promedio': np.random.normal(120, 40, n)
    })
    
    # Lógica de la variable objetivo: Acepta la promoción del aceite
    # Es más probable que acepten si gastan > 29.90 en abarrotes y usan tarjeta
    condicion_favorable = (df['Gasto_Abarrotes_Soles'] > 29.90) & (df['Uso_Tarjeta_Fidelidad'] == 1)
    probabilidades = np.where(condicion_favorable, 0.75, 0.20)
    df['Acepta_Promo_Aceite'] = np.random.binomial(1, probabilidades)
    
    return df

df = generar_datos()

# --- Menú Lateral (Fases CRISP-DM) ---
st.sidebar.title("Metodología CRISP-DM")
st.sidebar.info("Navega por las fases del proyecto de Machine Learning")
fase = st.sidebar.radio("Selecciona la fase:", [
    "1. Business Understanding",
    "2. Data Understanding",
    "3. Data Preparation",
    "4. Modeling",
    "5. Evaluation"
])

# --- FASE 1: Business Understanding ---
if fase == "1. Business Understanding":
    st.title("1. Comprensión del Negocio (Business Understanding)")
    st.markdown("""
    ### Objetivo
    Optimizar el perifoneo y la comunicación en tienda para una oferta específica: 
    **"Por compras mayores a S/29.90 en abarrotes, llévate a S/9.90 2 botellas de aceite de girasol"**.
    
    ### Problema de Machine Learning
    Crear un modelo de clasificación que prediga la probabilidad de que un cliente acepte esta promoción en la caja, basándose en su comportamiento de compra histórico y los artículos actuales en su carrito (categoría abarrotes).
    
    ### Criterio de Éxito
    * Aumentar la tasa de conversión de la promoción.
    * Reducir el tiempo de ofrecimiento en caja, enfocándose en clientes con alta probabilidad de compra.
    """)

# --- FASE 2: Data Understanding ---
elif fase == "2. Data Understanding":
    st.title("2. Comprensión de los Datos (Data Understanding)")
    
    st.subheader("Vista previa de las transacciones")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de la Variable Objetivo")
        fig_target = px.pie(df, names='Acepta_Promo_Aceite', hole=0.4, 
                            labels={'Acepta_Promo_Aceite': 'Aceptó Promo'},
                            color_discrete_sequence=['#ff9999','#66b3ff'])
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col2:
        st.subheader("Gasto en Abarrotes vs Aceptación")
        fig_scatter = px.box(df, x='Acepta_Promo_Aceite', y='Gasto_Abarrotes_Soles', 
                             color='Acepta_Promo_Aceite')
        # Línea de la regla de negocio
        fig_scatter.add_hline(y=29.90, line_dash="dot", annotation_text="Regla: S/29.90", annotation_position="bottom right")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- FASE 3: Data Preparation ---
elif fase == "3. Data Preparation":
    st.title("3. Preparación de los Datos (Data Preparation)")
    st.markdown("En esta fase separamos los datos en características (X) y la variable a predecir (y), y dividimos en conjuntos de entrenamiento y prueba.")
    
    X = df.drop('Acepta_Promo_Aceite', axis=1)
    y = df['Acepta_Promo_Aceite']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    col1, col2 = st.columns(2)
    col1.metric("Registros de Entrenamiento", X_train.shape[0])
    col2.metric("Registros de Prueba", X_test.shape[0])
    
    st.code("""
# Código de partición de datos
X = df.drop('Acepta_Promo_Aceite', axis=1)
y = df['Acepta_Promo_Aceite']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    """, language='python')

# --- FASE 4: Modeling ---
elif fase == "4. Modeling":
    st.title("4. Modelado (Modeling)")
    st.markdown("Entrenaremos un modelo de **Random Forest**, el cual es excelente para capturar reglas de negocio complejas en entornos de retail.")
    
    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando Random Forest..."):
            X = df.drop('Acepta_Promo_Aceite', axis=1)
            y = df['Acepta_Promo_Aceite']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            modelo.fit(X_train, y_train)
            
            # Guardamos el modelo en el estado de la sesión para usarlo en la siguiente fase
            st.session_state['modelo'] = modelo
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.success("¡Modelo entrenado exitosamente!")
            
            # Mostrar importancia de las variables
            importancia = pd.DataFrame({
                'Variable': X.columns,
                'Importancia': modelo.feature_importances_
            }).sort_values('Importancia', ascending=True)
            
            fig_imp = px.bar(importancia, x='Importancia', y='Variable', orientation='h',
                             title="¿Qué factores influyen más en la compra?")
            st.plotly_chart(fig_imp, use_container_width=True)

# --- FASE 5: Evaluation ---
elif fase == "5. Evaluation":
    st.title("5. Evaluación (Evaluation)")
    
    if 'modelo' not in st.session_state:
        st.warning("⚠️ Por favor, ve a la fase de 'Modeling' y entrena el modelo primero.")
    else:
        modelo = st.session_state['modelo']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.metric("Precisión (Accuracy) del Modelo", f"{acc * 100:.2f}%")
        
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicción", y="Realidad"),
                           x=['No Acepta', 'Acepta'], y=['No Acepta', 'Acepta'])
        st.plotly_chart(fig_cm)
        
        st.markdown("""
        **Conclusión de la Evaluación:**
        El modelo ha aprendido a identificar de forma eficiente qué perfil de cliente es propenso a llevarse la oferta del aceite. Este algoritmo puede integrarse en el sistema de punto de venta (POS) para emitir alertas automáticas al cajero.
        """)
