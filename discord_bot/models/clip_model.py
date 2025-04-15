import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPModelWrapper:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

    async def analyze(self, image, categories):
        inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        predicted_category = categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability
