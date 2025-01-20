import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def evaluate(encoder, projection, tasks):
    """
    Évalue le modèle sur toutes les tâches.
    """
    encoder.eval()
    projection.eval()

    all_predictions = []
    all_labels = []

    for task in tasks:
        dataloader = DataLoader(task, batch_size=32)
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                embeddings = encoder(images)
                predictions = projection(embeddings)
                predictions = torch.argmax(predictions, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy globale : {accuracy:.4f}")

def visualize_results():
    """
    Placeholder pour la visualisation des résultats (courbes, etc.).
    """
    pass