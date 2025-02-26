import numpy as np
import os
import torch
from PIL import Image
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


class Datasets(Dataset):
    def __init__(self, config, train=True):
        if train:
            print(config.train_data_dir)
            self.data_dir = config.train_data_dir
            _, self.im_height, self.im_width = config.image_dims
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((self.im_height, self.im_width)),
                    transforms.ToTensor()
                ]
            )
        else:
            self.data_dir = config.test_data_dir
            _, self.im_height, self.im_width = config.image_dims
            self.transform = transforms.Compose([

                    transforms.RandomCrop((self.im_height, self.im_width)),
                    transforms.ToTensor()
                ])
        self.imgs = []


        png_files = glob(os.path.join(self.data_dir, '*.png')) + glob(os.path.join(self.data_dir, '*.PNG'))
        self.imgs += png_files
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)
    
class testDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_cifar_loader(config):
    transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
    transform_test = transforms.Compose([
                transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=config.cifar_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root=config.cifar_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=0)
    return trainloader,testloader#drop_last = True

def get_trainloader(config):
    train_dataset = Datasets(config)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        batch_size=config.batch_size,
        worker_init_fn=worker_init_fn_seed,
        shuffle=False
    )
    return train_loader


def get_kodak_testloader(config):
    transform = transforms.Compose([
    transforms.RandomCrop(256),  # 随机裁剪图像
    transforms.ToTensor(),  # 将图像转换为张量
    ])

    test_dir = config.kodak_dir
    kodak_dataset = testDataset(test_dir, transform=transform)
    kodak_testloader = DataLoader(kodak_dataset, batch_size=2, shuffle=False)   
    return kodak_testloader

def get_clic_testloader(config):
    transform = transforms.Compose([
    transforms.CenterCrop(256),  # 随机裁剪图像
    transforms.ToTensor(),  # 将图像转换为张量
    ])
    
    test_dir = config.clic_dir
    clic_dataset = testDataset(test_dir, transform=transform)
    clic_testloader = DataLoader(clic_dataset, batch_size=2, shuffle=False)   
    return clic_testloader