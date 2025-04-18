from transformers import pipeline
import logging

class RoBERTaModelWrapper:
    def __init__(self, categories):
        self.categories = categories
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )

    async def analyze(self, text):
        result = self.classifier(text, candidate_labels=self.categories)

        results = []
        for label, score in zip(result["labels"], result["scores"]):
            logging.info(f"[RoBERTa] Probabilité pour {label} : {score:.2f}")
            results.append((label, score))

        return results