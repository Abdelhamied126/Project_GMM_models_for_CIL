from torchvision.datasets import ImageFolder
from torchvision import transforms
from train import train_linear_layer
from test_clip import test_with_clip
from model import MiniGPT4Wrapper
from config import CONFIG
import os

transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
])

def main():
    # Charger l'ensemble complet
    train_path = "/data/jahegoul10/cifar10_split/train"
    test_path = "/data/jahegoul10/cifar10_split/test"

    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # Filtrer pour ne garder que la classe 0
    class_idx = 0
    train_filtered = [(img, label) for img, label in train_dataset.samples if label == class_idx]
    test_filtered = [(img, label) for img, label in test_dataset.samples if label == class_idx]

    # Remplacer les datasets originaux par les datasets filtrés
    train_dataset.samples = train_filtered
    test_dataset.samples = test_filtered

    print(f"Number of training images for class {class_idx}: {len(train_dataset)}")
    print(f"Number of testing images for class {class_idx}: {len(test_dataset)}")

    # Entraînement sur l'ensemble d'entraînement filtré
    print(f"Training Task {class_idx}")
    train_linear_layer(train_dataset)

    print("Training complete. Proceeding to text generation...")
    # Génération des descriptions textuelles pour l'ensemble de test
    minigpt4 = MiniGPT4Wrapper(CONFIG["minigpt4_path"], CONFIG["vicuna7b_path"], CONFIG["device"])
    print("MiniGPT-4 model loaded successfully.")
    test_images = [img for img, _ in test_filtered]
    print("Test images loaded. Starting text generation...")
    generated_texts = [minigpt4.generate_text(None) for _ in test_images]  # Placeholder
    print("Text generation complete. Proceeding to CLIP testing...")

    # Testing avec CLIP sur l'ensemble de test
    print("Testing with CLIP...")
    test_with_clip(test_images, generated_texts)

if __name__ == "__main__":
    main()