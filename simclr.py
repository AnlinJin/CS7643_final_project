import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow_datasets.image_classification import Imagenette
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
from pl_bolts.models.self_supervised.resnets import resnet50

class ImageDataTransform:
    """
    Transform used for DataLoader
    output_length parameter depends on what model to use
    """

    def __init__(self, height=32, mode="train", output_length=3, no_aug=False):
        if no_aug:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                cifar10_normalization(),
            ])
        elif mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ])
        else: # val or test with aug
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

class SimCLR_finetuned(nn.Module):
    """
    Model for the downstream task
    Backbone will be initialized and trained if without feeding pretrained_model
    ref: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L42
    """

    def __init__(self, hidden_dim=2048, num_classes=10, pretrained_model=None):
        super(SimCLR_finetuned, self).__init__()
        if pretrained_model:
            self.backbone = pretrained_model.encoder
            self.backbone.requires_grad_(False)
        else: # just want to get encoder, parameters in SimCLR are not used
            self.backbone = resnet50()

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.backbone(x)[-1]  # bolts resnet returns a list
        out = self.linear_layer(out)
        return out

# 1st Stage
def cifar10_pretrain(num_epochs=300, batch_size=256):
    """
    Self-supervised pretraining, will not use test set
    """

    # Data Loader
    train_set = CIFAR10Dataset(mode="train", transform=ImageDataTransform(mode="train"))
    val_set = CIFAR10Dataset(mode="val", transform=ImageDataTransform(mode="val"))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

    # Model
    model = SimCLR(num_samples=40000, batch_size=batch_size, dataset="cifar10", gpus=1)

    # Fit
    trainer = Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# 2nd Stage
def cifar10_finetune(weight_path=None, num_epochs=100, batch_size=64, no_aug=False, log_file="./log_finetune", lr=0.001):
    """
    Hyper-parameters should be fixed across different self-supervised models
    ref: https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained model
    model = SimCLR.load_from_checkpoint(weight_path, strict=False) if weight_path else None
    model = SimCLR_finetuned(pretrained_model=model).to(device)

    # Data Loader
    train_set = CIFAR10Dataset(mode="train", transform=ImageDataTransform(mode="train", output_length=1, no_aug=no_aug))
    val_set = CIFAR10Dataset(mode="val", transform=ImageDataTransform(mode="val", output_length=1, no_aug=no_aug))
    test_set = CIFAR10Dataset(mode="test", transform=ImageDataTransform(mode="test", output_length=1, no_aug=no_aug))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Writer
    writer = SummaryWriter(log_file, flush_secs=30)

    # Training + Validation
    # ref: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    best_accuracy = 0.0
    best_model_path = None
    for epoch in range(1, num_epochs+1):

        # Training
        training_loss = 0.0  # per epoch
        with tqdm(train_loader) as t:
            t.set_description(f"Train {epoch}")

            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.to(device), labels.squeeze().to(device)

                # training
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.detach().cpu().numpy() / len(train_set)
        
        writer.add_scalar("Loss/training", training_loss, epoch)

        # Validation
        validation_loss = 0.0  # per epoch
        accuracy = 0.0
        with torch.no_grad():
            t = tqdm(val_loader)
            t.set_description(f"Valid {epoch}")

            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.to(device), labels.squeeze().to(device)

                # training
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                validation_loss += loss.detach().cpu().numpy() / len(val_set)

                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct_num = (predictions == labels).sum().item()
                accuracy += correct_num / len(val_set)
        
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Accuracy/validation", accuracy, epoch)

        # Update (for test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            print(f"deleting {best_model_path}")
            if best_model_path:
                os.remove(best_model_path)

            best_model_path = os.path.join(log_file, f"best_ckpt_e{epoch}.pth")
            print(f"saving {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

    # Test
    if not best_model_path:
        return
    best_model = SimCLR_finetuned().to(device)
    best_model.load_state_dict(torch.load(best_model_path), strict=False)

    accuracy = 0.0
    with torch.no_grad():
        t = tqdm(test_loader)
        t.set_description(f"Testing")

        for i, (inputs, labels) in enumerate(t):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = best_model(inputs)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct_num = (predictions == labels).sum().item()
            accuracy += correct_num / len(test_set)
    
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":

    pass
