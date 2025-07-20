import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from PIL import Image
import medmnist
from medmnist import INFO
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt

class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label[0]

class SimpleCNN(nn.Module):
    def __init__(self, num_kernels, kernel_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_kernels, kernel_size, padding=3)  # Ajuste no padding para manter o tamanho
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_kernels * 16 * 16, 128)  # Ajuste para dimensionar corretamente após pooling
        self.fc2 = nn.Linear(128, 8)  # Camada de saída para 8 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_and_prepare_data():
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_data = DataClass(split='train', download=True)
    val_data = DataClass(split='val', download=True)
    test_data = DataClass(split='test', download=True)

    images = np.concatenate((train_data.imgs, val_data.imgs, test_data.imgs), axis=0)
    labels = np.concatenate((train_data.labels, val_data.labels, test_data.labels), axis=0)

    total_images = images.shape[0]
    print(f"Total de imagens: {total_images}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = MedMNISTDataset(images, labels, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

def plot_misclassified_images(images, true_labels, predicted_labels, probabilities, class_names):
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Converte de (C, H, W) para (H, W, C)
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predicted_labels[i]]
        pred_prob = probabilities[i]
        title = f"V: {true_class}\nP: {pred_class}\nProb: {pred_prob:.2f}"
        plt.title(title)
        plt.axis('off')
    plt.show()

def train_and_evaluate_model(train_loader, val_loader, test_loader):
    model = SimpleCNN(16, 7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - Duration: {epoch_duration:.2f} seconds")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total training time: {total_duration:.2f} seconds")

    model.eval()
    all_labels = []
    all_preds = []
    all_images = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_images.extend(images)
            all_probs.extend(probs.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    class_names = ["Basófilos", "Eosinófilos", "Eritroblastos", "Granulócitos", "Linfócitos", "Monócitos", "Neutrófilos", "Plaqueta"]
    plot_confusion_matrix(cm, classes=class_names)

    # Identificar e plotar as imagens classificadas incorretamente
    misclassified_idxs = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    misclassified_images = [all_images[i] for i in misclassified_idxs[:5]]
    misclassified_true_labels = [all_labels[i] for i in misclassified_idxs[:5]]
    misclassified_pred_labels = [all_preds[i] for i in misclassified_idxs[:5]]
    misclassified_probs = [all_probs[i][all_preds[i]] for i in misclassified_idxs[:5]]
    plot_misclassified_images(misclassified_images, misclassified_true_labels, misclassified_pred_labels, misclassified_probs, class_names)

    return acc, cm

def main():
    train_loader, val_loader, test_loader = load_and_prepare_data()
    test_accuracy, confusion_matrix = train_and_evaluate_model(train_loader, val_loader, test_loader)
    print(f"Acurácia nos dados de teste: {test_accuracy * 100:.2f}%")
    print("Matriz de Confusão:")
    print(confusion_matrix)

if __name__ == '__main__':
    main()
