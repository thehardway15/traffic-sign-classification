import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print(torch.__version__)

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")

print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Selected device: {device}")

train_dir = './processed/Train'
test_dir = './processed'
test_label_csv = './trafic_sign_dataset/Test_data.csv'


class TrainDataset(Dataset):
    def __init__(self, img_dir, max_samples_per_class=None, transform=None, target_transform=None, device='cpu'):
        super(self.__class__, self).__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.device = device

        _filepaths = []
        _labels = []

        for label in os.listdir(train_dir):
            labelpath = os.path.join(train_dir, label)
            files = os.listdir(labelpath)
            if max_samples_per_class is not None:
                if len(files) > max_samples_per_class:
                    files = random.sample(files, max_samples_per_class)
            for f in files:
                _filepaths.append(f)
                _labels.append(label)

        _Fseries = pd.Series(_filepaths)
        _Lseries = pd.Series(_labels)
        self.img_labels = pd.concat([_Fseries, _Lseries], axis=1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 0])
        image = read_image(img_path).to(self.device)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, max_samples_per_class=None, transform=None, target_transform=None,
                 device='cpu'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        # Wczytanie danych z pliku CSV
        self.img_labels = pd.read_csv(csv_file)

        self.img_labels.columns = ['labels', 'path']
        self.img_labels['labels'] = self.img_labels['labels'].apply(lambda x: str(x))

        # Filtrujemy dane jeśli max_samples_per_class jest ustawione
        if max_samples_per_class is not None:
            # Grupujemy dane na podstawie etykiet i próbkujemy z każdej grupy
            self.img_labels = self.img_labels.groupby('labels', as_index=False).apply(
                lambda x: x.sample(min(len(x), max_samples_per_class))
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['path'])
        image = read_image(img_path).to(self.device)
        label = self.img_labels.iloc[idx]['labels']

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

target_transform = int
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float32),
])

dataset = TrainDataset(train_dir, 150, transform, target_transform, device=device)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
test_dataset = TestDataset(test_label_csv, test_dir, None, transform, target_transform, device=device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Wypisz informacje o wielkościach podziałów
print(f"Liczba rekordów w zbiorze treningowym: {len(train_dataset)}")
print(f"Liczba rekordów w zbiorze walidacyjnym: {len(val_dataset)}")
print(f"Liczba rekordów w zbiorze testowym: {len(test_dataset)}")


def analyze(model, test_loader, train_losses, valid_losses, train_accuracy, valid_accuracy, learning_rate):
    stats = open(f"./models/{learning_rate}/stats.txt", "w")

    print('Losses and accuracies chart')
    print('Loss train: ', train_losses[-1].detach().numpy())
    print('Loss valid: ', valid_losses[-1].detach().numpy())
    print('Accuracy and losses chart')
    print('Accuracy train: ', train_accuracy[-1].detach().numpy())
    print('Accuracy valid: ', valid_accuracy[-1].detach().numpy())

    stats.write(f"Loss train: {train_losses[-1].detach().numpy()}\n")
    stats.write(f"Loss valid: {valid_losses[-1].detach().numpy()}\n")
    stats.write(f"Accuracy train: {train_accuracy[-1].detach().numpy()}\n")
    stats.write(f"Accuracy valid: {valid_accuracy[-1].detach().numpy()}\n")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, valid_losses)

    plt.title('Average loss over time')
    plt.ylabel("loss/error")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Valid'])

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracy)), train_accuracy, valid_accuracy)
    plt.title('Accuracy over time')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Valid'])
    # plt.show()
    plt.savefig(f"./models/{learning_rate}/train_model.png")

    prediction_test = []
    correct_test = []
    for X_test, y_test in test_loader:
        X_test = X_test.to(torch.float32).to(device)
        y_test = y_test.to(torch.float32).to(device)

        y_pred = model.forward(X_test)

        _, predicted = torch.max(y_pred, 1)
        prediction_test.extend(predicted.tolist())
        correct_test.extend(y_test.tolist())

    print(f"Test accuracy: {accuracy_score(correct_test, prediction_test):.2f}")

    stats.write(f"Test accuracy: {accuracy_score(correct_test, prediction_test):.2f}\n")
    stats.close()


def train(model, train_loader, test_loader, valid_loader, epochs, loss_func, optim_func, learning_rate):
    t_losses = []
    v_losses = []
    t_accuracy = []
    v_accuracy = []

    local_model = model.to(device)

    for i in tqdm(range(epochs)):
        epoch_losses = []
        valid_losses_local = []
        train_accuracy_epoch = []
        valid_accuracy_epoch = []

        scaler = GradScaler()
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optim_func.zero_grad()

            with autocast():
                y_predict = local_model(X_train)
                loss = loss_func(y_predict, y_train)
            epoch_losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted_indexes = torch.max(y_predict, 1)
            train_accuracy_epoch.append(accuracy_score(predicted_indexes.cpu(), y_train.cpu()))

            # loss.backward()
            # optim_func.step()

        epoch_loss = torch.tensor(epoch_losses).mean(dtype=torch.float32)
        t_losses.append(epoch_loss)
        t_accuracy.append(torch.tensor(train_accuracy_epoch).mean(dtype=torch.float32))

        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)

                y_predict = local_model.forward(X_valid)
                loss = loss_func(y_predict, y_valid)

                valid_losses_local.append(loss.item())

                _, predicted_indexes = torch.max(y_predict, 1)
                valid_accuracy_epoch.append(accuracy_score(predicted_indexes.cpu(), y_valid.cpu()))

            valid_loss = torch.tensor(valid_losses_local).mean(dtype=torch.float32)
            v_losses.append(valid_loss)
            v_accuracy.append(torch.tensor(valid_accuracy_epoch).mean(dtype=torch.float32))

        # if i % 20 == 0 or i == epochs - 1:
        print(
            f'Epoch: {i}, Loss: {epoch_loss:.6f}, Valid: {valid_loss:.6f}, Accuracy train: {t_accuracy[-1]:.6f}, Accuracy valid: {v_accuracy[-1]:.6f}')


    if not os.path.exists(f"./models/{learning_rate}"):
        os.makedirs(f"./models/{learning_rate}")

    analyze(local_model, test_loader, t_losses, v_losses, t_accuracy, v_accuracy, learning_rate)

    torch.save(local_model.state_dict(), f"./models/{learning_rate}/model_{epochs}.pth")


class CnnNet(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()
        # Pierwsza warstwa konwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 128x128

        # Druga warstwa konwolucyjna
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 64x64

        # Trzecia warstwa konwolucyjna
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 32x32

        # Czwarta warstwa konwolucyjna
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 16x16

        # # Piąta warstwa konwolucyjna
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.relu5 = nn.ReLU()
        # self.pool5 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 8x8

        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()

        # Warstwy liniowe (FC)
        # self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        # x = self.pool5(self.relu5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


for i in range(10):
    print(f"Learning rate: {0.005 + (0.001 * i)}")
    net = CnnNet().to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005 + (0.001 * i)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    train(net, train_loader, test_loader, val_loader, 30, criterion, optimizer, learning_rate)
