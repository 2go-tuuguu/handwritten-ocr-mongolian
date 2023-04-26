import albumentations
import numpy as np
import torch
from PIL import Image, ImageOps

class MongolianOCRDataset:
    def __init__(self, image_paths, targets, resize=(300, 48)):
        self.image_paths = image_paths
        self.targets = targets

        self.resize = resize

        mean = (0,)
        std = (1,)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("L")
        image = ImageOps.expand(image, border=(0, 0, 0, self.resize[0]-image.size[1]), fill=255)
        image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
        targets = self.targets[item]

        image = ImageOps.invert(image)

        image = np.array(image)
        image = np.rot90(image) 

        image = np.expand_dims(image, axis=0).astype(np.float32)

        return {
            "images": torch.tensor(image).clone().detach().float(),
            "targets": targets.clone().detach().long(),
        }