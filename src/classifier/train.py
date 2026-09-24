import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

def get_data_loaders(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, 'Train')
    
    # Data Augmentation & Normalization
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    class_names = train_dataset.classes
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, class_names

def train_model(data_dir=".", num_epochs=10, batch_size=32, lr=0.001, save_path="models/resnet18_ripeness.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, class_names = get_data_loaders(data_dir, batch_size)
    print(f"Classes found: {class_names}")
    
    # Load Pretrained ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
