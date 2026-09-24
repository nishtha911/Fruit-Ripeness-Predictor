import os
import json
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

class RipenessPredictor:
    def __init__(self, checkpoint_path="models/resnet18_ripeness.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Reconstruct ResNet18 Model
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, default_fruit_type="Banana"):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        ripeness_stage = self.class_names[predicted_idx.item()]
        
        return {
            "fruit_type": default_fruit_type,
            "ripeness_stage": ripeness_stage,
            "confidence": round(confidence.item(), 2)
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predictor = RipenessPredictor()
        result = predictor.predict(img_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python predict.py <path_to_image>")
