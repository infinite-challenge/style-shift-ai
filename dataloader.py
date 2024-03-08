import os
from PIL import Image
from .model.sampler import InfiniteSampleWrapper
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import CIFAR10
from lightning import LightningDataModule
import pdb
from torch.utils import data

device = 'cpu'

class CustomImageDataset(Dataset):
    def __init__(
                self, 
                data_path : str,
                transform = None
                 ):
        super(CustomImageDataset, self).__init__()
        self.data_path = data_path
        
        if os.path.isdir(os.path.join(self.data_path, os.listdir(self.data_path)[0])):
            self.paths = []
            for file in os.listdir(self.data_path):
                for sub_file in os.listdir(os.path.join(self.data_path, file)):
                    self.paths.append(os.path.join(self.data_path, file, sub_file))
        else:
            self.paths = list(Path(self.data_path).glob("*"))
        
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(str(path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.paths)

def data_transform():
    transfrom_list = [
        transforms.Resize(size = (512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transfrom_list)

def imgtensor2pil(img_tensor):
    pil_img = transforms.ToPILImage()(img_tensor)
    return pil_img

class ImageDataModule(LightningDataModule):
    def __init__(self, content_data_path : str, style_data_path : str, batch_size : int = 4, num_workers : int = 8):
        super().__init__()
        self.content_data_path = content_data_path
        self.style_data_path = style_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_tf = data_transform()

    def setup(self, stage = None):
        self.content_dataset = CustomImageDataset(self.content_data_path, self.data_tf)
        self.style_dataset = CustomImageDataset(self.style_data_path, self.data_tf)

        self.content = data.DataLoader(
            self.content_dataset, batch_size = self.batch_size,
            sampler=InfiniteSampleWrapper(self.content_dataset), 
            num_workers=self.num_workers)
        self.style = data.DataLoader(
            self.style_dataset, batch_size = self.batch_size,
            sampler=InfiniteSampleWrapper(self.style_dataset), 
            num_workers=self.num_workers)

    def __len__(self):
        return 2 ** 31

    def train_dataloader(self):
        return zip(self.content, self.style)

    def val_dataloader(self):
        return zip(self.content, self.style)

    def test_dataloader(self):
        return zip(self.content, self.style)


# useage
# data_tf = data_transform()

# content_dataset = CustomImageDataset("./data/content", data_tf)
# style_dataset = CustomImageDataset("./data/style", data_tf)

# content = data.DataLoader(
#     content_dataset, batch_size = 8,
#     sampler=InfiniteSampleWrapper(content_dataset), 
#     num_workers=0)
# style = data.DataLoader(
#     style_dataset, batch_size = 8,
#     sampler=InfiniteSampleWrapper(style_dataset), 
#     num_workers=0)

# image_batch = zip(content, style)


class LitCIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size : int = 4, num_workers : int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


    def get_dataset(self, train, transform):
        dataset = CIFAR10(
            root=self.data_dir,
            train=train,
            transform=transform,
            download=train,
        )
        
        subset_indices = list(range(16))
        dataset = data.Subset(dataset, subset_indices)

        return dataset

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(size = (512, 512)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        dataset = self.get_dataset(
            train=True,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return dataloader, dataloader