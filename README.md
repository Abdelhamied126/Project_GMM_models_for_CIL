# Generative Multi-modal Models for Class-Incremental Learning (GMM for CIL)

## Introduction

Ce projet explore une approche innovante pour l’**apprentissage incrémental par classe (CIL)**, un défi en intelligence artificielle où les modèles doivent intégrer de nouvelles classes sans oublier celles précédemment apprises. Contrairement aux modèles discriminants traditionnels, notre solution utilise des **modèles génératifs multi-modaux (GMM)** pour réduire l’oubli catastrophique et améliorer l’efficacité.

## Objectifs

1. Proposer une méthode basée sur des GMM pour éviter l'oubli catastrophique.
2. Générer des descriptions textuelles détaillées pour la classification d'images.
3. Valider l'approche sur des ensembles de données standards comme **ImageNet-R** et **Tiny-ImageNet**.

## Méthodologie

### Architecture

1. **Encodeur Visuel (Vision Transformer)** :
   - Extrait des embeddings visuels riches des images.
2. **Décodeur Textuel (inspiré de MiniGPT-4)** :
   - Génère des descriptions textuelles telles que : "Ceci est une photo d’un chien".
3. **Projection Linéaire** :
   - Aligne les espaces visuel et textuel pour une intégration fluide des nouvelles classes.
4. **CLIP** :
   - Compare les descriptions textuelles générées avec les classes existantes dans un espace latent partagé.

### Avantages

- **Pas de modification de l’architecture** pour ajouter de nouvelles classes.
- **Réduction de l’oubli** grâce à l’utilisation des descriptions textuelles.
- **Efficacité computationnelle** : seul le module de projection est optimisé.

## Résultats
![image](https://github.com/user-attachments/assets/f8e76c4c-28ae-4e92-93dc-075347c9df37)


- **ImgR-alin** :
  - Précision moyenne : **93.38 %** sur 8 tâches.
    
## Applications

- **Systèmes embarqués** :
  - Reconnaissance faciale ou d'objets évolutive pour caméras intelligentes.
- **Surveillance et sécurité** :
  - Intégration continue de nouvelles menaces ou activités anormales.
- **Santé** :
  - Classification d’images médicales avec ajout progressif de nouvelles pathologies.

#### Pour plus de détails consulter : 
`[Projet_3A.pdf](./Projet_3A.pdf)`


