import torch
from transformers import AutoImageProcessor, ResNetForImageClassification


class ResNetModelWrapper:
    def __init__(self):
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)

    async def analyze(self, image, categories):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability

