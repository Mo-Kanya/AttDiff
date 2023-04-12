from torch.utils import data
from functools import partial
from PIL import Image
from pathlib import Path

import torchvision
from torchvision import transforms

# Dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image


def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return transforms.functional.resize(image, min_size)
    return image

# pass --dataset_name MNIST to cli.py to automatically sample from MNIST
# It will oversample the 'true' digit correctly

class MNIST_1vA(torch.utils.data.Dataset):
    def __init__(self, folder='./', digit=8):
        self.image_size = 32  # TODO: No hardcoding.

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        self.dataset = torchvision.datasets.MNIST(folder, train=True, download=True, transform=self.transform)
        self.dataset.targets = self.dataset.targets == digit

    def __getlen__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image

    def __len__(self):
        return len(self.dataset)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent=False, aug_prob=0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# Augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., types=[], detach=False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

def get_dataloaders(data_folder, dataset_name, cfg):
    if dataset_name is None:
        dataset = Dataset(data_folder, cfg.image_size, transparent=cfg.transparent,
                                aug_prob=cfg.dataset_aug_prob)

        dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, 
                                    shuffle=True, drop_last=True, pin_memory=True)

    if dataset_name == 'MNIST':
        dataset = MNIST_1vA(digit=8)

        weights = make_weights_for_balanced_classes(dataset.dataset, cfg.num_classes)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)

    loader = cycle(dataloader)

    # auto set augmentation prob for user if dataset is detected to be low
    num_samples = len(dataset)
    if not exists(cfg.aug_prob) and num_samples < 1e5:
        cfg.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
        print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')
    
    return loader
