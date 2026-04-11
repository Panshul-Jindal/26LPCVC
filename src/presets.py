import torch
from torchvision.transforms import transforms
from torchvision.transforms import transforms


class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        hflip_prob=0.5,
    ):
        trans = [
            transforms.ConvertImageDtype(torch.float32),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([transforms.RandomCrop(crop_size)])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, *, crop_size):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.CenterCrop(crop_size),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
