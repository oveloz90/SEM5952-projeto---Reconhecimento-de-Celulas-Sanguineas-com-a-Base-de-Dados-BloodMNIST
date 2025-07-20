import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import pandas as pd
from PIL import Image
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import seaborn as sns
import time

class_names = ["Basófilos", "Eosinófilos", "Eritroblastos", "Granulócitos", "Linfócitos", "Monócitos", "Neutrófilos",
               "Plaquetas"]


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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        identity_downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4))
        layers = []
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
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
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = MedMNISTDataset(images, labels, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset = DataLoader(torch.utils.data.Subset(dataset, range(train_size)), batch_size=32, shuffle=True)
    val_dataset = DataLoader(torch.utils.data.Subset(dataset, range(train_size, train_size + val_size)), batch_size=32,
                             shuffle=False)
    test_dataset = DataLoader(torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset))),
                              batch_size=32, shuffle=False)
    return train_dataset, val_dataset, test_dataset


def display_incorrect_samples(test_loader, model, device, num_samples=5):
    incorrects = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            for i in range(len(images)):
                if preds[i] != labels[i]:
                    incorrects.append((images[i], labels[i], preds[i], probs[i][preds[i]].item()))
                if len(incorrects) >= num_samples:
                    break
            if len(incorrects) >= num_samples:
                break
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for ax, (img, true_label, pred_label, prob) in zip(axes, incorrects):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.set_title(
            f'Verdadeiro: {class_names[true_label.item()]}\nPredito: {class_names[pred_label.item()]}\nProb: {prob:.2f}')
        ax.axis('off')
    plt.show()


def plot_classification_report(report, class_names):
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            report_data.append({
                'class': label,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })
    report_df = pd.DataFrame.from_records(report_data)
    report_df.set_index('class', inplace=True)

    plt.figure(figsize=(12, 7))
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('Relatório de Classificação para Teste')
    plt.show()


def train_and_evaluate_model(train_loader, val_loader, test_loader, device):
    model = ResNet(Block, [3, 4, 6, 3], 3, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()  # End timer
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Acurácia Global: {acc * 100:.2f}%")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    plot_classification_report(report, class_names)

    display_incorrect_samples(test_loader, model, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_and_prepare_data()
    train_and_evaluate_model(train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
