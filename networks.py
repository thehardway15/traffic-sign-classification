import torch.nn as nn


class CnnNet(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()
        # Pierwsza warstwa konwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 112x112

        # Druga warstwa konwolucyjna
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 56x56

        # Trzecia warstwa konwolucyjna
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 28x18

        # Czwarta warstwa konwolucyjna
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 14x14

        # Piąta warstwa konwolucyjna
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 7x7

        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()

        # Warstwy liniowe (FC)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CnnNetV2(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Redukuje wymiar do 112x112
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Redukuje wymiar do 56x56
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Redukuje wymiar do 28x18
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Redukuje wymiar do 14x14
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Redukuje wymiar do 7x7
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CnnNetV3(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()
        # Pierwsza warstwa konwolucyjna
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 112x112
        )

        # Druga warstwa konwolucyjna
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 56x56
        )

        # Trzecia warstwa konwolucyjna
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 28x18
        )

        # Czwarta warstwa konwolucyjna
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 14x14
        )

        # Piąta warstwa konwolucyjna
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 7x7
        )

        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()

        # Warstwy liniowe (FC)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CnnNetV4(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()
        # Pierwsza warstwa konwolucyjna
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 112x112
        )

        # Druga warstwa konwolucyjna
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 56x56
        )

        # Trzecia warstwa konwolucyjna
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 28x18
        )

        # Czwarta warstwa konwolucyjna
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 14x14
        )

        # Piąta warstwa konwolucyjna
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 7x7
        )

        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()

        # Warstwy liniowe (FC)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CnnNetV5(nn.Module):
    def __init__(self, num_classes=205):
        super().__init__()
        # Pierwsza warstwa konwolucyjna
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 112x112
        )

        # Druga warstwa konwolucyjna
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 56x56
        )

        # Trzecia warstwa konwolucyjna
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 28x18
        )

        # Czwarta warstwa konwolucyjna
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 14x14
        )

        # Piąta warstwa konwolucyjna
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Redukuje wymiar do 7x7
        )

        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()

        # Warstwy liniowe (FC)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x