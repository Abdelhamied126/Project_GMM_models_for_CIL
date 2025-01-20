import os
import shutil
from sklearn.model_selection import train_test_split
from config import CONFIG

def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Sépare le dataset en ensembles d'entraînement, de validation et de test.
    Crée automatiquement les répertoires nécessaires.

    Args:
    - dataset_path : Chemin du dataset original (avec des sous-dossiers par classe).
    - output_path : Chemin pour stocker les ensembles divisés.
    - train_ratio : Ratio d'entraînement (par défaut 70%).
    - val_ratio : Ratio de validation (par défaut 15%).
    - test_ratio : Ratio de test (par défaut 15%).
    """
    # Vérification des ratios
    assert train_ratio + val_ratio + test_ratio == 1.0, "Les ratios doivent totaliser 1.0"

    # Créer les répertoires de sortie pour train, val et test
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    # Parcourir chaque classe dans le dataset original
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        # Obtenir toutes les images de la classe
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith((".png", ".jpg"))]

        # Diviser les images en train, val, et test
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        # Copier les fichiers dans les répertoires correspondants
        for split, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_path, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)  # Crée le dossier pour cette classe
            for img in img_list:
                shutil.copy(img, os.path.join(split_class_dir, os.path.basename(img)))

    print(f"Dataset split completed. Data saved in: {output_path}")


if __name__ == "__main__":
    # Chemin vers le dataset original
    dataset_path = CONFIG["dataset_path"]  # Exemple : "/data/jahegoul10/cifar10_classes"

    # Chemin pour stocker les ensembles divisés
    output_path = CONFIG["split_dataset_path"]  # Exemple : "/data/jahegoul10/cifar10_split"

    # Diviser le dataset
    split_dataset(dataset_path, output_path)