import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms import transforms

class MnistDataset(data.Dataset):
    def __init__(self, root="data/MNIST", phase="train", imsize=128, old_output_map={}):
        super(MnistDataset, self).__init__()
        self.phase = phase
        self.obtain_old_outputs = False
        self.transform = transforms.Compose(
                    [
                        transforms.Resize((imsize, imsize)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (1.0,))
                    ]
                )  

        self.images = [os.path.join(root, image) for image in open(os.path.join(root, f"mnist_{self.phase}.txt")).read().splitlines()]
        self.targets = [int(img.split('/')[-2]) for img in self.images] 
        self.names = [img.split('/')[-1].replace('.png', '') for img in self.images]
        self.old_output_map = old_output_map
        
    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img_name = self.names[index]
        img = Image.open(img)
        
        if self.transform is not None:
            img = self.transform(img)

        # Use for training of LwF
        if len(self.old_output_map):
            old_output = self.old_output_map[img_name]
            return img, target, old_output  
        
        # Use for get name - old task output mapping
        if self.obtain_old_outputs:
            return img_name, img, target    
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_names(self):
        return len(self.names)
