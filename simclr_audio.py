import os
import numpy as np
import torch
import torch.nn as nn
import torchvision, torchaudio
import torch_audiomentations
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm.contrib import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from pytorch_lightning.trainer.trainer import Trainer
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet50


class AudioDataTransform:
    """
    Transform used for DataLoader
    output_length parameter depends on what model to use
    """

    def __init__(self, height=32, mode="train", output_length=3):
        if mode == "train":
            self.transforms = transforms.Compose([
                torch_audiomentations.Gain(),
                torch_audiomentations.PolarityInversion(),
                torch_audiomentations.AddColoredNoise(max_snr_in_db=15, sample_rate=16000),
                torchaudio.transforms.Spectrogram(normalized=True),
                torchaudio.transforms.AmplitudeToDB(), # if not used, spectrogram becomes unreadable
                torchvision.transforms.RandomCrop(size=(201, 116)), # minimum size
            ])
        else: # val or test
            self.transforms = transforms.Compose([
                torchaudio.transforms.Spectrogram(normalized=True),
                torchaudio.transforms.AmplitudeToDB(), # if not used, spectrogram becomes unreadable
                torchvision.transforms.RandomCrop(size=(201, 116)), # minimum size
            ])
        
        self.output_length = output_length

    def __call__(self, sample):
        # output_length=1 is for supervised training
        if self.output_length == 1:
            return self.transforms(sample)
        
        # output_length=3 is for SimCLR pre-training
        return [self.transforms(sample) for _ in range(self.output_length)]

class LibrispeechDataset(Dataset):
    """
    Create train/val/test split for Librispeech Dataset
    train: 2303, val: 200, test: 200, num_labels: 40
    """
    def __init__(self, mode="train", transform=None, path="./librispeech"):
        # get data from path
        dataset = torchaudio.datasets.LIBRISPEECH("./librispeech", url="dev-clean", download=True)
        ids, ys = list(range(len(dataset))), [speaker_id for _, _, _, speaker_id, _, _ in dataset]

        # train/val/test split
        id_train, id_test, y_train, y_test = train_test_split(ids, ys, test_size=200, random_state=0, stratify=ys)
        id_train, id_val, y_train, y_val = train_test_split(id_train, y_train, test_size=200, random_state=0, stratify=y_train)
        
        # attributes
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.id_train, self.id_val, self.id_test = id_train, id_val, id_test
        self.id_to_label = {y: i for i, y in enumerate(set(ys))} # for training indexing

    def __len__(self):
        if self.mode == "train":
            return len(self.id_train)
        return len(self.id_val) if self.mode == "val" else len(self.id_test)

    def __getitem__(self, idx):
        if self.mode == "train":
            new_idx = self.id_train[idx]
        else:
            new_idx = self.id_val[idx] if self.mode == "val" else self.id_test[idx]

        audio, _, _, speaker_id, _, _ = self.dataset[new_idx]
        label = self.id_to_label[speaker_id]

        if self.transform:
            audio = audio.unsqueeze(0) # required channel dimenstion for torch_audiomentations
            audio = self.transform(audio)
            if type(audio) == list: # when transform's output_length > 1
                audio = [w.squeeze(0) for w in audio]
            else:
                audio = audio.squeeze(0) # change back
        
        return audio, label

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
            self.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.backbone(x)[-1]  # bolts resnet returns a list
        out = self.linear_layer(out)
        return out

# 1st Stage
def librispeech_pretrain(num_epochs=300, batch_size=64):
    """
    Self-supervised pretraining, will not use test set
    """

    # Data Loader
    train_set = LibrispeechDataset(mode="train", transform=AudioDataTransform(mode="train"))
    val_set = LibrispeechDataset(mode="val", transform=AudioDataTransform(mode="val"))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

    # Model
    model = SimCLR(num_samples=len(train_set), batch_size=batch_size, dataset="", gpus=1)
    model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn

    # Fit
    trainer = Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# 2nd Stage
def librispeech_finetune(weight_path=None, num_epochs=100, batch_size=64, log_file="./log_finetune", lr=0.001):
    """
    Hyper-parameters should be fixed across different self-supervised models
    ref: https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained model
    if weight_path:
        model = SimCLR(num_samples=-1, batch_size=-1, dataset="", gpus=1)
        model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn
        model.load_state_dict(torch.load(weight_path), strict=False)
    else:
        model = None

    model = SimCLR_finetuned(pretrained_model=model, num_classes=40).to(device)

    # Data Loader
    train_set = LibrispeechDataset(mode="train", transform=AudioDataTransform(mode="train", output_length=1))
    val_set = LibrispeechDataset(mode="val", transform=AudioDataTransform(mode="val", output_length=1))
    test_set = LibrispeechDataset(mode="test", transform=AudioDataTransform(mode="test", output_length=1))

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
    best_model = SimCLR_finetuned(num_classes=40).to(device)
    best_model.load_state_dict(torch.load(best_model_path), strict=False)

    accuracy = 0.0
    with torch.no_grad():
        t = tqdm(test_loader)
        t.set_description(f"Testing")

        for i, (inputs, labels) in enumerate(t):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct_num = (predictions == labels).sum().item()
            accuracy += correct_num / len(test_set)
    
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":

    librispeech_pretrain()