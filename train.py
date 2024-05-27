import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler

from dataloaders import TrainDataset, TestDataset
from train_loop import train_loop
from networks import CnnNetV3, CnnNet, CnnNetV4, CnnNetV5

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

# print(torch.__version__)

# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device: {torch.cuda.current_device()}")

# print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# print(f"Selected device: {device}")

train_dir = './processed/Train'
test_dir = './processed'
test_label_csv = './trafic_sign_dataset/Test_data.csv'

target_transform = int
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = TrainDataset(train_dir, 170, transform, target_transform, device=device)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
test_dataset = TestDataset(test_label_csv, test_dir, None, transform, target_transform, device=device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Wypisz informacje o wielkościach podziałów
# print(f"Liczba rekordów w zbiorze treningowym: {len(train_dataset)}")
# print(f"Liczba rekordów w zbiorze walidacyjnym: {len(val_dataset)}")
# print(f"Liczba rekordów w zbiorze testowym: {len(test_dataset)}")

dataset_remap = TrainDataset(train_dir, 11000, transform, target_transform, device=device, remap_label=True, excludes=['6'])
train_dataset_remap, val_dataset_remap = random_split(dataset_remap, [0.8, 0.2])
test_dataset_remap = TestDataset(test_label_csv, test_dir, 6000, transform, target_transform, device=device, remap_label=True, excludes=['6'])

train_loader_remap = DataLoader(train_dataset_remap, batch_size=64, shuffle=True)
test_loader_remap = DataLoader(test_dataset_remap, batch_size=64, shuffle=False)
val_loader_remap = DataLoader(val_dataset_remap, batch_size=64, shuffle=False)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    net = CnnNetV5(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loop(net, train_loader_remap, val_loader_remap, test_loader_remap, optimizer, criterion, 200, device, "models/v7", True, scheduler, True, 0.01)
