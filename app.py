import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détection de Masques",
    page_icon="🎭",
    layout="wide"
)

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
@st.cache_resource
def load_model():
    model = YOLO('mask_detection_model.pt')
    return model

model = load_model()

# Classes
CLASSES = {
    0: "😷 Avec Masque",
    1: "😊 Sans Masque",
    2: "⚠️ Masque Mal Porté"
}

# ============================================================
# INTERFACE
# ============================================================
st.title("🎭 Détection de Masques Faciaux")
st.markdown("Uploadez une image pour détecter automatiquement le port du masque avec YOLO v8")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05)
    
    st.markdown("---")
    st.markdown("### 🤖 À propos")
    st.markdown("""
    - **Modèle** : YOLOv8n
    - **Dataset** : Face Mask Detection
    - **Classes** : 3
    """)

# Upload d'image
uploaded_file = st.file_uploader(
    "📸 Choisissez une image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Afficher l'image originale
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)
    
    # Bouton d'analyse
    if st.button("🔍 Analyser", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Prédiction
            results = model.predict(
                source=image,
                conf=confidence,
                iou=0.45,
                verbose=False
            )
            
            # Image annotée
            annotated_image = results[0].plot()
            
            # Statistiques
            detections = results[0].boxes
            num_detections = len(detections)
            
            # Compter par classe
            stats = {
                "😷 Avec Masque": 0,
                "😊 Sans Masque": 0,
                "⚠️ Masque Mal Porté": 0
            }
            
            if num_detections > 0:
                for box in detections:
                    class_id = int(box.cls[0])
                    stats[CLASSES[class_id]] += 1
            
            # Afficher le résultat
            with col2:
                st.subheader("Résultat de la détection")
                st.image(annotated_image, use_container_width=True)
            
            # Statistiques
            st.markdown("---")
            st.subheader("📊 Statistiques")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total détecté", num_detections)
            with metric_cols[1]:
                st.metric("Avec masque", stats["😷 Avec Masque"])
            with metric_cols[2]:
                st.metric("Sans masque", stats["😊 Sans Masque"])
            with metric_cols[3]:
                st.metric("Mal porté", stats["⚠️ Masque Mal Porté"])

else:
    st.info("👆 Uploadez une image pour commencer")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<p>Créé avec ❤️ | Propulsé par Streamlit & YOLOv8</p>"
    "</div>",
    unsafe_allow_html=True
)
