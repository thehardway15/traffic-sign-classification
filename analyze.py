import os
import torch
from networks import *
from sklearn.metrics import confusion_matrix
from dataloaders import TestDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision.utils import make_grid

test_dir = './processed'
test_label_csv = './trafic_sign_dataset/Test_data.csv'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

target_transform = int
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# test_dataset = TestDataset(test_label_csv, test_dir, 20, transform, target_transform, device=device)
test_dataset = TestDataset(test_label_csv, test_dir, 6000, transform, target_transform, device=device, remap_label=True, excludes=['6'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Przeskaluj z powrotem do [0, 1]
    return tensor


def analyze(klass, version, num_classes=205):
    model = klass(num_classes=num_classes)
    model.load_state_dict(torch.load(f'./models/{version}/last/model.pt', map_location='cpu'))
    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred.extend(output.argmax(1).tolist())
            y_true.extend(target.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')

    with open(f'./models/{version}/accuracy.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}')

    #cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    #cm.figure_.savefig(f'./models/{version}/cm.png')
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./models/{version}/cm.png')
    plt.close()

    index = 0
    missclassified = []
    for label, predict in zip(y_true, y_pred):
        if label != predict:
            missclassified.append(index)
        index += 1

    # show missclassified sample images
    plt.figure(figsize=(20, 10))
    for plotIndex, idx in enumerate(missclassified[:20]):
        plt.subplot(4, 5, plotIndex+1)
        img = test_dataset[idx][0].cpu()
        img = unnormalize(img)
        plt.imshow(make_grid(img).permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Predicted: {y_pred[idx]}, True: {y_true[idx]}')
    plt.savefig(f'./models/{version}/missclassified.png')
    plt.close()

if __name__ == '__main__':
    analyze(CnnNetV5, 'v7', 6)
    # analyze(CnnNetV5, 'v6')
    # analyze(CnnNetV5, 'v5')
    # analyze(CnnNetV4, 'v4')
    # analyze(CnnNetV3, 'v3')
    # analyze(CnnNetV2, 'v2')
    # analyze(CnnNet, 'v1')
