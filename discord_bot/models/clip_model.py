from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

class CLIPModelWrapper:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        print("[INFO] [CLIP] Modèle CLIP chargé avec succès.")

    async def analyze(self, image, categories):
        try:
            if not categories:
                print("[ERROR] [CLIP] Liste des catégories vide.")
                return None, 0.0

            if isinstance(image, str):
                if image.startswith("http"):
                    image = Image.open(BytesIO(requests.get(image).content)).convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")

            # Log des catégories
            print(f"[INFO] [CLIP] Catégories : {categories}")
            inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = outputs.logits_per_image.softmax(dim=1).squeeze()

            # Log des probabilités
            print(f"[INFO] [CLIP] Probabilités : {probs}")

            # Log des probabilités arrondies pour chaque catégorie
            for i, category in enumerate(categories):
                print(f"[INFO] [CLIP] Probabilité pour {category} : {probs[i].item():.2f}")

            predicted_category = categories[probs.argmax()]
            predicted_probability = probs.max().item()

            # Log du résultat final
            print(f"[INFO] [CLIP] Catégorie prédite : {predicted_category} avec probabilité {predicted_probability:.2f}")
            return predicted_category, predicted_probability

        except Exception as e:
            print(f"[ERROR] [CLIP] Erreur analyse image : {e}")
            return None, 0.0