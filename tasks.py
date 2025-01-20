from torch.utils.data import Subset
import os
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# Transformation pour charger les données
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adapter la taille pour BLIP ou CLIP
    transforms.ToTensor(),
])

# Charger CIFAR-10
cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

def split_by_class(dataset, num_classes, output_dir="./class_datasets"):
    """
    Sépare un dataset global en sous-datasets contenant une seule classe chacun.

    Args:
    - dataset : Dataset global (CIFAR, Tiny ImageNet, etc.)
    - num_classes : Nombre total de classes dans le dataset.
    - output_dir : Dossier où stocker les sous-datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()  # Pour convertir un tenseur en image PIL

    for class_idx in range(num_classes):
        # Trouver les indices des exemples correspondant à la classe
        indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]

        # Créer un sous-dataset
        class_subset = Subset(dataset, indices)

        # Sauvegarder les données
        class_dir = os.path.join(output_dir, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)

        for i, (image, label) in enumerate(class_subset):
            image = to_pil(image)  # Convertir le tenseur en image PIL
            image_path = os.path.join(class_dir, f"image_{i}.png")
            image.save(image_path)

        print(f"Classe {class_idx} sauvegardée avec {len(indices)} images dans {class_dir}")

# Exécuter la séparation des classes
split_by_class(cifar10, num_classes=10, output_dir="../cifar10_classes")