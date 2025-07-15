import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),  # output size dopo conv layers e' [64,5,5]
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_mnist(epochs: int, save_path: str = "outputs/mnist_model.pth"):
    os.makedirs("outputs", exist_ok=True)  # 🔹 Crea la cartella se non esiste

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{(epoch + 1)}/{epochs}], loss: {running_loss/len(train_loader):.4f}')
    
    print(f'Training time: {time.time() - start:.2f} seconds')
    torch.save(model.state_dict(), save_path)  # 🔹 Salva il modello in outputs/

def evaluate_mnist(model_path: str = "outputs/mnist_model.pth"):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    assert os.path.exists(model_path), f"Model file {model_path} does not exist."

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    print(f'\nTest accuracy: {accuracy:.4f}')

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    os.makedirs("outputs", exist_ok=True)  # 🔹 Assicurati che la cartella esista
    plt.savefig("outputs/confusion_matrix.png")  # 🔹 Salva nella cartella outputs
    plt.show(block=False)

if __name__ == "__main__":
    train_mnist(5)
    evaluate_mnist()
