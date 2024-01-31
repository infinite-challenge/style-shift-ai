import os
from PIL import Image
from model.sampler import InfiniteSampleWrapper
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset

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

data_tf = data_transform()

content_dataset = CustomImageDataset("./data/content", data_tf)
style_dataset = CustomImageDataset("./data/style", data_tf)

content = data.DataLoader(
    content_dataset, batch_size = 8,
    sampler=InfiniteSampleWrapper(content_dataset), 
    num_workers=0)
style = data.DataLoader(
    style_dataset, batch_size = 8,
    sampler=InfiniteSampleWrapper(style_dataset), 
    num_workers=0)

image_batch = zip(content, style)


