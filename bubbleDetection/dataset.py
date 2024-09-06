from PIL import Image
import numpy as np
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os import path
import torchvision

def load_image(idx, transform = None):
    filename = f"data/image/haima/keyframe-{str(idx).zfill(3)}.png"
    img = Image.open(filename)
    img = np.array(img)
    if transform: img = transform(img)
    return img

class BubbleDataset(Dataset):
    def __init__(self, train=False, test=False, valid=False, classes = 4):
        if train: 
            self.imgs_path = "data/image/haima_bubble_dataset/train"
        elif test:
            self.imgs_path = "data/image/haima_bubble_dataset/test"
        elif valid:
            self.imgs_path = "data/image/haima_bubble_dataset/valid"
        self.label = pd.read_csv(path.join(self.imgs_path, 'label.csv'))
        self.img_dim = (64, 64, 3)
        self.img_size = 64
        self.classes = classes

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        img_idx = self.label["index"][idx]
        img = Image.open(path.join(self.imgs_path, f"{str(img_idx).zfill(6)}.png")) 

        transforms = [torchvision.transforms.ToTensor()] 
        width = height = self.img_size 
        scale = min(width / img.width, height / img.height) 
        new_width, new_height = int(img.width * scale), int(img.height * scale)
        diff_width, diff_height = width - new_width, height - new_height
        resize = torchvision.transforms.Resize(size=(new_height, new_width))
        pad = torchvision.transforms.Pad(
            padding=(
                diff_width // 2,
                diff_height // 2,
                diff_width // 2 + diff_width % 2,
                diff_height // 2 + diff_height % 2,
            )
        )
        transforms = [resize, pad] + transforms
        transformation = torchvision.transforms.Compose(transforms)
        img_tensor = transformation(img)

        class_id = self.label["bubble_number"][idx]
        return img_tensor, class_id

if __name__ == "__main__": 
    dataset = BubbleDataset(test=True) 
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # img, label = next(iter(data_loader))

    for imgs, labels in data_loader: 
        print("Batch of images has shape: ",imgs.shape)
        print("Batch of labels has shape: ", labels.shape)
