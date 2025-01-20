import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split

def create_tasks(dataset_path, num_tasks=3):
    """
    Divise un dataset en plusieurs tâches (une classe par tâche).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    classes_per_task = len(dataset.classes) // num_tasks

    tasks = []
    for i in range(num_tasks):
        start_idx = i * classes_per_task
        end_idx = start_idx + classes_per_task
        task_classes = dataset.classes[start_idx:end_idx]
        task_data = [(x, y) for x, y in dataset if y in range(start_idx, end_idx)]

        task_dataset = torch.utils.data.TensorDataset(*zip(*task_data))
        tasks.append(task_dataset)

    return tasks