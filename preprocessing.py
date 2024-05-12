import os
import torch
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image


transform = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((256, 256)),  # lub (200, 200) w zależności od wyboru
    v2.PILToTensor(),
    v2.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def preprocessing_train():
    train_dir = './trafic_sign_dataset/train'
    output_dir = './processed/Train'

    for _, directory in tqdm(enumerate(os.listdir(train_dir))):
        if os.path.isdir(os.path.join(train_dir, directory)):
            for image in os.listdir(os.path.join(train_dir, directory)):
                img = Image.open(os.path.join(train_dir, directory, image))
                img = transform(img)
                pil_img = v2.ToPILImage()(img)
                if not os.path.exists(os.path.join(output_dir, directory)):
                    os.makedirs(os.path.join(output_dir, directory))
                pil_img.save(os.path.join(output_dir, directory, image))


def preprocessing_test():
    test_dir = './trafic_sign_dataset/test'
    output_dir = './processed/Test'

    for _, filename in tqdm(enumerate(os.listdir(test_dir))):
        img = Image.open(os.path.join(test_dir, filename))
        img = transform(img)
        pil_img = v2.ToPILImage()(img)
        pil_img.save(os.path.join(output_dir, filename))



if __name__ == '__main__':
    preprocessing_train()
    preprocessing_test()