import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import medmnist
from medmnist import INFO
import pandas as pd
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
        self.conv1 = nn.Conv2d(3, num_kernels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_kernels * 16 * 16, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

def get_flattened_size(model, input_shape):
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        x = model.conv1(x)
        x = model.relu(x)
        x = model.pool(x)
        x = model.flatten(x)
    return x.shape[1]

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

def train_and_evaluate_cnn(num_kernels, kernel_size):
    train_loader, val_loader, test_loader = load_and_prepare_data()
    model = SimpleCNN(num_kernels, kernel_size)

    # Determinar o tamanho correto da camada linear
    input_shape = (3, 32, 32)
    flattened_size = get_flattened_size(model, input_shape)
    model.fc1 = nn.Linear(flattened_size, 8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Avaliação nos dados de validação
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch + 1}, Val Accuracy: {val_acc:.4f}')

    return val_accuracies

def save_results_to_excel(results, filename='cnn_results.xlsx'):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)

def main():
    num_kernels_list = [8, 16, 32]
    kernel_size_list = [3, 5, 7]
    results = []

    for num_kernels in num_kernels_list:
        fig, ax = plt.subplots()
        for kernel_size in kernel_size_list:
            print(f'\nAvaliação com {num_kernels} kernels e tamanho de kernel {kernel_size}')
            val_accuracies = train_and_evaluate_cnn(num_kernels, kernel_size)
            ax.plot(val_accuracies, label=f'Kernel size {kernel_size}')
            results.append({
                'num_kernels': num_kernels,
                'kernel_size': kernel_size,
                'val_accuracies': val_accuracies
            })
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Acurácia de Validação')
        ax.set_title(f'Acurácia de Validação para {num_kernels} Kernels')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    save_results_to_excel(results)

if __name__ == '__main__':
    main()
