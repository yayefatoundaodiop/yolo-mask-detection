import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
print("🔄 Chargement du modèle...")
model = YOLO('mask_detection_model.pt')
print("✅ Modèle chargé avec succès !")

# Classes
CLASSES = {
    0: "😷 Avec Masque",
    1: "😊 Sans Masque",
    2: "⚠️ Masque Mal Porté"
}

# ============================================================
# FONCTION DE DÉTECTION
# ============================================================
def detect_masks(image):
    """
    Détecte les masques sur une image
    
    Args:
        image: Image PIL ou numpy array
    
    Returns:
        Image annotée avec les détections
    """
    if image is None:
        return None
    
    # Prédiction
    results = model.predict(
        source=image,
        conf=0.25,  # Seuil de confiance
        iou=0.45,   # Seuil IOU pour NMS
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
    
    # Texte de résumé
    summary = f"""
    📊 **Résultats de la détection :**
    
    - Personnes détectées : **{num_detections}**
    - {stats['😷 Avec Masque']} avec masque
    - {stats['😊 Sans Masque']} sans masque  
    - {stats['⚠️ Masque Mal Porté']} avec masque mal porté
    """
    
    return annotated_image, summary

# ============================================================
# INTERFACE GRADIO
# ============================================================

# CSS personnalisé
custom_css = """
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

#description {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 20px;
}

.gradio-container {
    max-width: 1200px;
    margin: auto;
}
"""

# Exemples d'images (vous pouvez ajouter vos propres URLs)
examples = [
    # Ajoutez ici des chemins vers des images d'exemple
]

# Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("<h1 id='title'>🎭 Détection de Masques Faciaux</h1>")
    gr.HTML("<p id='description'>Uploadez une image pour détecter automatiquement le port du masque avec YOLO v8</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                label="📸 Image d'entrée",
                type="pil",
                height=400
            )
            
            # Boutons
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyser", variant="primary", size="lg")
                clear_btn = gr.ClearButton(components=[input_image], value="🔄 Effacer")
        
        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(
                label="✨ Résultat de la détection",
                type="numpy",
                height=400
            )
            
            # Statistiques
            output_text = gr.Markdown(label="📊 Statistiques")
    
    # Informations
    with gr.Accordion("ℹ️ À propos", open=False):
        gr.Markdown("""
        ### 🎯 Comment ça marche ?
        
        1. **Uploadez** une image contenant une ou plusieurs personnes
        2. Cliquez sur **"Analyser"**
        3. Le modèle YOLO détecte automatiquement :
           - ✅ Les personnes avec masque
           - ❌ Les personnes sans masque
           - ⚠️ Les masques mal portés
        
        ### 🤖 Technologie
        
        - **Modèle** : YOLOv8n (Ultralytics)
        - **Entraînement** : Face Mask Detection Dataset
        - **Classes** : with_mask, without_mask, mask_weared_incorrect
        - **Framework** : Gradio + Hugging Face Spaces
        
        ### 📊 Performance
        
        - mAP@50 : ~0.75-0.85
        - Temps d'inférence : <100ms par image
        - Support : Images JPG, PNG
        
        ### 🔗 Liens utiles
        
        - [Code source sur GitHub](#)
        - [Dataset sur Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
        - [Documentation YOLO](https://docs.ultralytics.com)
        """)
    
    # Exemples
    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_image,
            outputs=[output_image, output_text],
            fn=detect_masks,
            cache_examples=True
        )
    
    # Event handlers
    submit_btn.click(
        fn=detect_masks,
        inputs=input_image,
        outputs=[output_image, output_text]
    )
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; margin-top: 20px; color: #666;'>
        <p>Créé avec ❤️ par [Votre Nom] | Propulsé par Gradio & YOLOv8</p>
    </div>
    """)

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    demo.launch(
        share=False,  # Sur HF Spaces, pas besoin de share=True
        show_error=True
    )
