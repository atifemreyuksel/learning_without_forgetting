import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import transforms

class ImagenetDataset(data.Dataset):
    def __init__(self, root="data/imagenet", phase="val", imsize=224):
        super(ImagenetDataset, self).__init__()
        if phase != "val":
            raise NotImplementedError("This data loader is written for only validation dataset!")
        self.phase = phase
        # Transform block reference -> https://github.com/pytorch/vision/issues/39
        self.transform = transforms.Compose(
                    [
                        transforms.Resize((imsize, imsize)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ]
                )  
        self.images = sorted(glob(os.path.join(root, "images", "*.JPEG")))
        # Please retrieve gts from https://github.com/BVLC/caffe/tree/master/data/ilsvrc12
        self.targets = [int(label.split(' ')[1]) for label in open(os.path.join(root, "gts", f"{self.phase}.txt")).read().splitlines()] 
        assert len(self.images) == len(self.targets), "Size of images and targets must be equal! Please check."
        
    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)
