import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
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
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

def plot_classification_report(cr, title='Relatório de Classificação', cmap='RdYlGn'):
    df_cr = pd.DataFrame(cr).T
    sns.heatmap(df_cr.iloc[:-1, :].drop(['support'], axis=1), annot=True, cmap=cmap)
    plt.title(title)
    plt.show()

def train_and_evaluate_model(train_loader, val_loader, test_loader):
    input_size = 32 * 32 * 3  # RGB images of 32x32
    hidden_size = 128
    num_classes = 8
    model = SimpleMLP(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=["Basófilos", "Eosinófilos", "Eritroblastos", "Granulócitos", "Linfócitos", "Monócitos", "Neutrófilos", "Plaquetas"], output_dict=True)

    plot_confusion_matrix(cm, classes=["Basófilos", "Eosinófilos", "Eritroblastos", "Granulócitos", "Linfócitos", "Monócitos", "Neutrófilos", "Plaquetas"])
    plot_classification_report(cr, title='Relatório de Classificação para Teste')

    return acc, cm, cr

def main():
    train_loader, val_loader, test_loader = load_and_prepare_data()
    test_accuracy, confusion_matrix, classification_report = train_and_evaluate_model(train_loader, val_loader, test_loader)
    print(f"Acurácia nos dados de teste: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
