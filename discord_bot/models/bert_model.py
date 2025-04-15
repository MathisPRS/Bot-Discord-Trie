import torch
from transformers import BertForSequenceClassification, BertTokenizer


class BERTModelWrapper:
    def __init__(self, categories):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))
        self.categories = categories

    async def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability
