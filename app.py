import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Digit Recognizer")
st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro.")

# 1. Cargar el modelo guardado
# Según la teoría de la Presentacion IA unidad 5 sesion 12.pdf, 
# cargamos el modelo en RAM/VRAM para responder peticiones.
@st.cache_resource
def load_my_model():
    # Asegúrate de que el nombre del archivo sea exacto al que guardaste
    return tf.keras.models.load_model('modelo_mnist.keras')

model = load_my_model()

# 2. Crear el lienzo (Canvas) para dibujar
canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=20,
    stroke_color="white",
    background_color="black", 
    height=280, 
    width=280,
    drawing_mode="freedraw", 
    key="canvas",
)

# 3. Procesar el dibujo y predecir
# El bloque 'if' debe contener todo el proceso de transformación e inferencia
if canvas_result.image_data is not None:
    # Convertir el dibujo a 28x28 píxeles (formato MNIST)
    # Se corrige la asignación de 'img' que estaba cortada
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    
    # Siguiendo la práctica: conversión a escala de grises y normalización
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0 # Normalizar datos (0 a 1)

    # Predicción (Inferencia)
    # Redimensionamos para que coincida con la entrada del modelo (1, 28, 28, 1)
    pred = model.predict(img.reshape(1, 28, 28, 1))
    clase = np.argmax(pred)
    confianza = np.max(pred)

    # 4. Mostrar resultados con Umbral de Seguridad
    # Aplicamos la lógica de negocio explicada en la teoría[cite: 1]
    st.subheader(f"Resultado: {clase}")
    
    if confianza < 0.80:
        st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
    else:
        st.success(f"Confianza alta: {confianza:.2%}")
        # Visualización de probabilidades mediante un gráfico de barras
        st.bar_chart(pred[0])
