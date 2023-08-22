import torch
from torch import nn
from torchvision.transforms import (
    CenterCrop,
    ConvertImageDtype,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision.transforms.functional import InterpolationMode


class ImageTransform(nn.Module):
    def __init__(self, img_size, is_train=True):
        super(ImageTransform, self).__init__()
        self.is_train = is_train

        self.train_transforms = nn.Sequential(
            Resize([img_size], interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            CenterCrop(img_size),
            ConvertImageDtype(torch.float16),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.test_transforms = nn.Sequential(
            Resize([img_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(img_size),
            ConvertImageDtype(torch.float16),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __call__(self, x) -> torch.Tensor:
        with torch.no_grad():
            if self.is_train:
                x = self.train_transforms(x)
            else:
                x = self.test_transforms(x)
        return x
