import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from config import CONFIG
from model import MiniGPT4Wrapper
import os

def train_linear_layer(task_data):
    device = CONFIG["device"]
    model = MiniGPT4Wrapper(CONFIG["minigpt4_path"], CONFIG["vicuna7b_path"], device)
    linear_layer = nn.Linear(768, CONFIG["num_classes"]).to(device)  # Couche linéaire de classification

    optimizer = optim.Adam(linear_layer.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(task_data, batch_size=CONFIG["batch_size"], shuffle=True)

    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Simuler les embeddings via MiniGPT-4
            image_embedding = torch.rand((images.size(0), 768)).to(device)  # Placeholder pour des embeddings réels
            outputs = linear_layer(image_embedding)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader)}")

    # Créer le répertoire si nécessaire
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder les poids de la couche linéaire
    torch.save(linear_layer.state_dict(), os.path.join(output_dir, "linear_layer.pth"))