from PIL import Image
import numpy as np
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os import path
import torchvision
import os 
import torchvision.transforms.functional as F
import json

def load_image(idx, transform = None):
    filename = f"data/image/haima/keyframe-{str(idx).zfill(3)}.png"
    img = Image.open(filename)
    img = np.array(img)
    if transform: img = transform(img)
    return img

# ------------ Haima Bubble dataset -----------------------
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

# ------------------- DAVIS 2017 dataset -------------------
class DAVIS2017(Dataset):
    """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 db_root_dir='data/DAVIS/',
                 transform=None,
                 seq_name=None,
                 pad_mirroring=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            inputRes (tuple): image size after reshape (HEIGHT, WIDTH)
            db_root_dir (path): path to the DAVIS2017 dataset
            transform: set of Albumentation transformations to be performed with A.Compose
            meanval (tuple): set of magic weights used for normalization (np.subtract(im, meanval))
            seq_name (str): name of a class: i.e. if "bear" one im of "bear" class will be retrieved
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.seq_name = seq_name
        self.pad_mirroring = pad_mirroring

        if self.train==1:
            fname = 'train'
        elif self.train==0:
            fname = 'val'
        else:
            fname = "test-dev"

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            # even though we could avoid using the txt files, we might have to use them
            # due to consistency: maybe some sub-folders shouldn't be included and we know which
            # to consider in the .txt file only
            with open(os.path.join(db_root_dir, "ImageSets/2017", fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    # why sort? And are we using np.sort cause we need the data-structure to be np.array
                    # instead of a list? Maybe it's faster
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    # why using lambda? map applies a given function to each item of an iterable. Apparently
                    # lambda here has two purposes: 1) makes the os.path.join a function as first arg of map()
                    # 2) provides an argument x for os.path.join(root_folder, sub_folder, x=image)
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    # here we're creating a list of all the path to the images
                    img_list.extend(images_path)
                    # same thing for the labels
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)

                    # what if we wanted to create the labels for a simple classification task?
                    #lab = [seq.strip() for i in range(len(os.listdir(os.path.join
                    #      (db_root_dir, "Annotations/Full-Resolution", seq.strip()))))]
                    #labels.extend(lab)
        else:

            # retrieves just one img and mask of a specified class (seq_name)
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))

            img_list = list(map(lambda x: os.path.join('JPEGImages/Full-Resolution/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]

            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]
                
        print(len(labels), len(img_list))
        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # idx order is controlled by torch.utils.data.DataLoader(shuffle): if shuffle = True, idx will be
        # retrieved randomly, otherwise they will be sequential from 0 to __len__
        img = np.array(Image.open(os.path.join(self.db_root_dir, self.img_list[idx])).convert("RGB"), dtype=np.float32)
        gt = np.array(Image.open(os.path.join(self.db_root_dir, self.labels[idx])).convert("L"), dtype=np.float32)
        
        gt = ((gt/np.max([gt.max(), 1e-8])) > 0.5).astype(np.float32)
        
        if self.transform is not None:

            augmentations = self.transform(image=img, mask=gt)
            img = augmentations["image"]
            gt = augmentations["mask"]
            
        # if image width and height is < than expected shape --> we should apply mirroring:
        # with padding_mode="reflect"
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.Pad.html
        if self.pad_mirroring:
            img = F.Pad(padding=self.pad_mirroring, padding_mode="reflect")(img)

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        return list(img.shape[:2])



if __name__ == "__main__": 
    # dataset = BubbleDataset(test=True) 
    dataset = BubbleDotLabel()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # img, label = next(iter(data_loader))

    for imgs, labels in data_loader: 
        print("Batch of images has shape: ",imgs.shape)
        print("Batch of labels has shape: ", labels.shape)
