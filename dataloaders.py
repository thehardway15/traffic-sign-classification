from torchvision.io import read_image
from torch.utils.data import Dataset
import random
import pandas as pd
import os


class_group_dict = {
    '0' : ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '14', '15', '16', '17', '32', '41', '42','46','72', '73', '74', '75', '76', '77',
          '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '140', '141',
          '144', '145'],
    '1' : ['11', '13', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '44', '45', '48', '49', '53', '54', '55',
           '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69'],
    '2' : ['12', '27', '43', '47', '110', '111', '112', '113', '114', '115', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
          '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '142', '143','148', '152', '153', '154', '155', '156', '158',
          '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182'],
    '3' : ['33', '34', '35', '36', '37', '38', '39', '40', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '146', '147',
          '149', '150', '151'],
    '4' : ['70', '71', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202',
          '203', '204'],
    '5' : ['116', '138', '139', '157', '159', '160', '161', '162', '163', '164'],
    '6': ['50', '51', '52'],
}

class_name_dict = {
    '0' : 'prohibition',
    '1' : 'warning',
    '2' : 'information',
    '3' : 'mandatory',
    '4' : 'supplements',
    '5' : 'cities',
    '6': 'trains',
}


def find_new_class(class_group_dict, old_class):
    for key, values in class_group_dict.items():
        if old_class in values:
            return key
    return None


class TrainDataset(Dataset):
    def __init__(self, img_dir, max_samples_per_class=None, transform=None, target_transform=None, device='cpu', remap_label=False, excludes=[]):
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
            for f in files:
                _label = label
                if remap_label:
                    _label = find_new_class(class_group_dict, label)
                if _label in excludes:
                    continue

                if max_samples_per_class is not None and len(list(filter(lambda x: x == _label, _labels))) >= max_samples_per_class:
                    continue
                _filepaths.append(os.path.join(label, f))
                _labels.append(_label)

        _Fseries = pd.Series(_filepaths)
        _Lseries = pd.Series(_labels)
        self.img_labels = pd.concat([_Fseries, _Lseries], axis=1)

    def get_table(self):
        return self.img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).to(self.device)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, max_samples_per_class=None, transform=None, target_transform=None,
                 device='cpu', remap_label=False, excludes=[]):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        # Wczytanie danych z pliku CSV
        self.img_labels = pd.read_csv(csv_file)

        self.img_labels.columns = ['labels', 'path']
        self.img_labels['labels'] = self.img_labels['labels'].apply(lambda x: str(x))

        if remap_label:
            self.img_labels['labels'] = self.img_labels['labels'].apply(lambda x: find_new_class(class_group_dict, x))

        if excludes:
            self.img_labels = self.img_labels[~self.img_labels['labels'].isin(excludes)]

        # Filtrujemy dane jeśli max_samples_per_class jest ustawione
        if max_samples_per_class is not None:
            # Grupujemy dane na podstawie etykiet i próbkujemy z każdej grupy
            self.img_labels = self.img_labels.groupby('labels', as_index=False).apply(
                lambda x: x.sample(min(len(x), max_samples_per_class))
            ).reset_index(drop=True)

    def get_table(self):
        return self.img_labels

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
