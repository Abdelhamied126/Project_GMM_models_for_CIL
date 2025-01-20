import torch

CONFIG = {
    "dataset_path": "/data/jahegoul10/codes/data/cifar100_classes",  # Dossier contenant les classes originales
    "split_dataset_path": "/data/jahegoul10/codes/data/cifar100_split",  # Dossier pour les ensembles divisés
    "output_dir": "./results",  # Dossier de sortie pour les modèles entraînés
    "minigpt4_path": "/data/jahegoul10/vicuna-7b/prerained_minigpt4_7b.pth",  # Chemin vers MiniGPT-4
    "vicuna7b_path": "/data/jahegoul10/vicuna-7b",  # Chemin vers Vicuna
    "clip_model_path": "openai/clip-vit-base-patch32",  # Modèle CLIP
    "image_size": 224,
    "batch_size": 16,
    "epochs": 5,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "huggingface_cache": "/data/jahegoul10/huggingface_cache",
    "num_classes": 10,
    "num_tasks": 10  # Nombre total de tâches (par exemple, 10 pour CIFAR-100)
}