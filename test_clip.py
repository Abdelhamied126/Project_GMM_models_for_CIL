import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import CONFIG
import os

# Redéfinir le chemin de cache Hugging Face
os.environ["HF_HOME"] = "/data/jahegoul10/huggingface_cache"

print(os.environ["HF_HOME"])  # Devrait afficher "/data/jahegoul10/huggingface_cache"

def test_with_clip(image_paths, generated_texts):
    device = CONFIG["device"]
    clip_model = CLIPModel.from_pretrained(CONFIG["clip_model_path"]).to(device)
    processor = CLIPProcessor.from_pretrained(CONFIG["clip_model_path"])

    # Texte à comparer
    class_descriptions = ["This is a plane", "This is a car", "This is a bird", "This is a cat",
                          "This is a deer", "This is a dog", "This is a frog", "This is a horse",
                          "This is a ship", "This is a truck"]

    correct = 0
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=class_descriptions, images=image, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
        predicted_class = outputs.logits_per_image.argmax().item()
        print(f"Image {i}: Predicted class = {predicted_class}")
        if predicted_class == 0:  # La classe correcte est "plane" (classe 0)
            correct += 1

    print(f"Accuracy: {correct / len(image_paths) * 100:.2f}%")