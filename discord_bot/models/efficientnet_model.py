import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetModelWrapper:
    def __init__(self):
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.eval()

    async def analyze(self, image, categories):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        predicted_category = categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability