from torchvision.io import read_image
from torch.utils.data import Dataset
import random
import pandas as pd
import os


class TrainDataset(Dataset):
    def __init__(self, img_dir, max_samples_per_class=None, transform=None, target_transform=None, device='cpu'):
        super(self.__class__, self).__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.device = device

        _filepaths = []
        _labels = []

        for label in os.listdir(img_dir):
            labelpath = os.path.join(img_dir, label)
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

