import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os


transform = transforms.Compose([
    transforms.Resize((300, 300)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),  
    transforms.ToTensor(),  
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])


dataset_path = "/content/drive/MyDrive/train"


dataset = datasets.ImageFolder(root=dataset_path, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class_names = dataset.classes
print(f"Classes: {class_names}")

test_size

import torch.nn as nn
import torchvision.models as models


model = models.resnet50(pretrained=True)


num_classes = len(class_names)  
model.fc = nn.Linear(in_features=2048, out_features=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 30

best_loss = float("inf")
best_model_path = "/content/drive/MyDrive/train/best_car_brand_classifier.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

      
        outputs = model(images)
        loss = criterion(outputs, labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved with loss {best_loss:.4f}")

model.load_state_dict(torch.load("/content/drive/MyDrive/train/best_car_brand_classifier.pth"))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

from sklearn.metrics import classification_report

y_true = []
y_pred = []


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


print(classification_report(y_true, y_pred, target_names=class_names))

