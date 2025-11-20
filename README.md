# Traffic Sign Detection

Un système de détection et classification de panneaux de signalisation routière utilisant l'apprentissage profond et la vision par ordinateur.

## Technologies utilisées

- **Python** - Langage principal
- **OpenCV** - Traitement d'images et vision par ordinateur
- **TensorFlow/Keras** - Réseaux de neurones convolutifs
- **NumPy** - Calculs scientifiques
- **Matplotlib** - Visualisation des résultats

## Structure du projet

traffic-sign-detection/
├── data_processing.py # Traitement d'images et détection des panneaux
├── detection.py # Classification et affichage des résultats
├── model.py # Architecture et entraînement du modèle CNN
├── main.py # Script principal d'exécution
├── requirements.txt # Dépendances Python
└── data/ # Dataset des panneaux (à télécharger séparément)

## Fonctionnalités

- **Détection des panneaux** par analyse de couleur (rouge) et morphologie
- **Classification** en 44 classes différentes de panneaux
- **Modèle CNN** avec architecture optimisée pour la reconnaissance
- **Interface visuelle** avec bounding boxes et labels

## Performance

- Modèle CNN avec couches convolutionnelles et fully-connected
- Prétraitement avancé des images (HSV, flou, dilatation)
- Détection robuste grâce aux contraintes de taille et forme

## Installation et utilisation

1. Cloner le repository
2. Installer les dépendances : `pip install -r requirements.txt`
3. Télécharger le dataset dans le dossier `data/`
4. Exécuter : `python main.py`

## Notes

Ce projet est conçu pour le dataset Belgian Traffic Sign et peut être adapté pour d'autres datasets de signalisation routière.
