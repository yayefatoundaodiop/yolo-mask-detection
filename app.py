import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détecteur de Masques", page_icon="😷")

st.title("😷 Détecteur de Masques - YOLOv8")
st.write("Le déploiement est réussi ! Chargez une image pour tester votre modèle.")

# Chargement du modèle (avec ton vrai nom de fichier)
@st.cache_resource
def load_model():
    return YOLO("mon_modele_final.pt")

model = load_model()

# Interface de téléchargement
uploaded_file = st.file_uploader("Choisissez une photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file)
    
    # Bouton pour lancer la détection
    if st.button('Lancer la détection'):
        with st.spinner('Analyse en cours...'):
            # Prédiction
            results = model(image)
            
            # Récupérer l'image avec les boîtes tracées
            res_plotted = results[0].plot()
            
            # Affichage
            st.image(res_plotted, caption='Résultat de la détection', use_container_width=True)
            
            # Petit message de succès
            st.success("Détection terminée !")
