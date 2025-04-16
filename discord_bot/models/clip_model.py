from io import BytesIO
import requests
import torch
import logging
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
                return []

            if isinstance(image, str):
                if image.startswith("http"):
                    image = Image.open(BytesIO(requests.get(image).content)).convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")

            inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = outputs.logits_per_image.softmax(dim=1).squeeze()

            # Log des probabilités arrondies pour chaque catégorie
            results = []
            for i, category in enumerate(categories):
                probability = probs[i].item()
                logging.info(f"[CLIP] Probabilité pour {category} : {probability:.2f}")
                results.append((category, probability))

            return results

        except Exception as e:
            logging.error(f"[CLIP] Erreur analyse image : {e}")
            return []
