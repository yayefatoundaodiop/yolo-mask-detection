---
title: Détection de Masques Faciaux
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Détection de Masques Faciaux avec YOLO

Cette application détecte automatiquement le port du masque facial sur des images grâce à un modèle YOLOv8 entraîné sur le dataset Face Mask Detection.

## 🎯 Fonctionnalités

- ✅ Détection des personnes **avec masque**
- ❌ Détection des personnes **sans masque**
- ⚠️ Détection des **masques mal portés**
- 📊 Statistiques en temps réel
- 🚀 Interface intuitive et responsive

## 🤖 Modèle

- **Architecture** : YOLOv8n (nano)
- **Dataset** : [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Classes** : 3 (with_mask, without_mask, mask_weared_incorrect)
- **Performance** : mAP@50 ~0.80

## 🛠️ Technologies

- **Framework ML** : Ultralytics YOLOv8
- **Interface** : Gradio
- **Hosting** : Hugging Face Spaces

## 📝 Utilisation

1. Uploadez une image
2. Cliquez sur "Analyser"
3. Visualisez les détections et les statistiques

## 🔗 Liens

- [Code source](#)
- [Dataset original](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [Documentation YOLO](https://docs.ultralytics.com)

## 📜 License

MIT License - Libre d'utilisation

---

Créé avec ❤️ par [Votre Nom]
