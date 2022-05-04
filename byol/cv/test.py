import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm.contrib import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from pytorch_lightning.trainer.trainer import Trainer
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur
from pl_bolts.models.self_supervised import BYOL

class ImageDataTransform:
    """
    Transform used for DataLoader
    output_length parameter depends on what model to use
    """

    def __init__(self, height=32, mode="train", output_length=3):
        if mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ])
        else: # val or test
            self.transforms = transforms.Compose([
                transforms.Resize(height + 12),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                cifar10_normalization(),
            ])
        
        self.output_length = output_length

    def __call__(self, sample):
        # output_length=1 is for supervised training
        if self.output_length == 1:
            return self.transforms(sample)
        
        # output_length=3 is for SimCLR pre-training
        return [self.transforms(sample) for _ in range(self.output_length)]


class CIFAR10Dataset(Dataset):
    """
    Create train/val/test split for CIFAR10 Dataset
    """
    def __init__(self, mode="train", transform=None):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                    test_size=5000, random_state=0, stratify=y_train)
        
        if mode == "train":
            self.images, self.labels = X_train, y_train
        elif mode == "val":
            self.images, self.labels = X_val, y_val
        else:
            self.images, self.labels = X_test, y_test

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = Image.fromarray(image) # for torchvision.transforms.ToTensor()
        if self.transform: # note: ImageDataTransform will produce a tuple
            image = self.transform(image) 
        
        return image, label


class BYOL_finetuned(nn.Module):
    """
    Model for the downstream task
    Backbone will be initialized and trained if without feeding pretrained_model
    ref: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L42
    """

    def __init__(self, num_classes=10, pretrained_model=None):
        super(BYOL_finetuned, self).__init__()
        if pretrained_model:
            self.backbone = pretrained_model.online_network.encoder
            self.backbone.requires_grad_(False)
        else: # just want to get encoder, parameters in SimCLR are not used
            self.backbone = BYOL(num_classes=10, batch_size=256, dataset="cifar10", gpus=1).online_network.encoder

        self.linear_layer = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.backbone(x)[-1]  # bolts resnet returns a list
        out = self.linear_layer(out)
        return out

if __name__ == "__main__":
    # Data Loader
    batch_size = 64
    test_set = CIFAR10Dataset(mode="test", transform=ImageDataTransform(mode="test", output_length=1))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #best_model_path = "./log_finetune/version_7/best_ckpt_e254.pth"
    best_model_path = "./log_finetune/version_8/best_ckpt_e96.pth"
    best_model = BYOL_finetuned().to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    accuracy = 0.0
    # top_1_score = []
    # top_2_score = []
    num = 5
    top_scores = [[] for _ in range(num)]
    with torch.no_grad():
        t = tqdm(test_loader)
        t.set_description(f"Testing")

        for i, (inputs, labels) in enumerate(t):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = best_model(inputs)
            
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            # outputs_soft = nn.functional.softmax(outputs, dim=1)
            # values, _ = torch.topk(outputs_soft, 2, dim=1)
            # top_1_score = np.append(top_1_score, [e for e in values[:, 0].cpu().detach().numpy()])
            # top_2_score = np.append(top_2_score, [e for e in values[:, 1].cpu().detach().numpy()])
            outputs_soft = nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
            for o in outputs_soft:
                o = sorted(o, reverse=True)
                for i in range(num):
                    top_scores[i].append(o[i])
            correct_num = (predictions == labels).sum().item()
            accuracy += correct_num / len(test_set)

    for i in range(num):
        print(f"Top-{i+1} Mean: {np.mean(top_scores[i])}")
        print(f"Top-{i+1} std: {np.std(top_scores[i])}")

    #top_1_score = [x.cpu().detach().numpy() for x in top_1_score]
    # print(top_1_score)
    print(f"Test Accuracy: {accuracy}")
#     print(f"Top-1 Mean: {np.mean(top_1_score)}")
#     print(f"Top-1 std: {np.std(top_1_score)}")

#     print(f"Top-2 Mean: {np.mean(top_2_score)}")
#     print(f"Top-2 std: {np.std(top_2_score)}")

