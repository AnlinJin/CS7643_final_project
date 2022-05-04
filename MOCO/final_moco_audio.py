import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchaudio
import torch_audiomentations
import tensorflow as tf

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tqdm.contrib import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from pytorch_lightning.trainer.trainer import Trainer
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet50
from pl_bolts.models.self_supervised import Moco_v2
from torch.nn import functional as F
from pl_bolts.metrics import mean, precision_at_k


class AudioDataTransform:
    """
    Transform used for DataLoader
    output_length parameter depends on what model to use
    """

    def __init__(self, height=32, mode="train", output_length=3, no_aug=False):
        if no_aug:
            self.transforms = transforms.Compose([
                torchaudio.transforms.Spectrogram(normalized=True),
                torchaudio.transforms.AmplitudeToDB(), # if not used, spectrogram becomes unreadable
            ])
        elif mode == "train":
            self.transforms = transforms.Compose([
                torch_audiomentations.Gain(),
                torch_audiomentations.PolarityInversion(),
                torch_audiomentations.AddColoredNoise(max_snr_in_db=15, sample_rate=16000),
                torchaudio.transforms.Spectrogram(normalized=True),
                torchaudio.transforms.AmplitudeToDB(), # if not used, spectrogram becomes unreadable
                torchvision.transforms.RandomCrop(size=(201, 116)), # minimum size
            ])
        else: # val or test with aug
            self.transforms = transforms.Compose([
                torchaudio.transforms.Spectrogram(normalized=True),
                torchaudio.transforms.AmplitudeToDB(), # if not used, spectrogram becomes unreadable
                torchvision.transforms.RandomCrop(size=(201, 116)), # minimum size
            ])
        
        self.output_length = output_length

    def __call__(self, inp):
        q = self.transforms(inp)
        k = self.transforms(inp)
        return q, k

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
            audio = audio.unsqueeze(1) # required channel dimenstion for torch_audiomentations
            q, k = self.transform(audio)
            q = q.squeeze(1)
            k = k.squeeze(1)

        return (q, k), label

class Moco_new(Moco_v2):
    """
    Model for the downstream task to avoid the dataset check
    ref: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py#L256
    """
    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), labels = batch

        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

class Moco_finetuned(nn.Module):
    """
    Model for the downstream task
    Backbone will be initialized and trained if without feeding pretrained_model
    ref: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L42
    """

    def __init__(self, hidden_dim=128, num_classes=10, pretrained_model=None):
        super(Moco_finetuned, self).__init__()
        if pretrained_model:
            self.backbone = pretrained_model.encoder_q
            self.backbone.requires_grad_(False)
        else: # just want to get encoder, parameters in SimCLR are not used
            self.backbone = resnet50()
            self.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.backbone(x)  # bolts resnet returns a list
        out = self.linear_layer(out)
        return out

# 1st Stage
def librispeech_pretrain(num_epochs=300, batch_size=128):
    """
    Self-supervised pretraining, will not use test set
    """
    # Data Loader
    train_set = LibrispeechDataset(mode="train", transform=AudioDataTransform(mode="train"))
    val_set = LibrispeechDataset(mode="val", transform=AudioDataTransform(mode="val"))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, drop_last = True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6, drop_last = True)

    # Data Loader
    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Moco_new.add_model_specific_args(parser)
    args = parser.parse_args()


    # Model
    model = Moco_new(**args.__dict__)
    model.encoder_q.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn
    model.encoder_k.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # mono-chn

    # Fit
    trainer = Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":

    librispeech_pretrain()