import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import namegenerator

from tqdm import tqdm
from typing import Tuple
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import random_split

from kiss.sampler.base import Sampler
from kiss.utils.strings import Format
from kiss.utils.configs import CONFIGS


class Experiment:
    def __init__(self,
                 model: Module,
                 dataset_tr: VisionDataset,
                 dataset_te: VisionDataset,
                 sampler_cls: Sampler,
                 ratio: float | Tuple[float, float, int],
                 name: str = None,
                 **kwargs):

        self.model_ = model
        self.dataset_tr_ = dataset_tr
        self.dataset_te_ = dataset_te

        if isinstance(ratio, float):
            self.ratio = ratio

        if isinstance(ratio, tuple):
            self.rstart = ratio[0]
            self.rstop = ratio[1]
            self.rnum = ratio[2]

        self.sampler_cls = sampler_cls
        
        self.experiment_name = name or f"{self.model_.__class__.__name__}-{self.dataset_tr_.__class__.__name__}"
        
        self.batch_size = kwargs.get("batch_size") or 64
        self.num_workers = kwargs.get("num_workers") or 0
        self.epochs = kwargs.get("epochs") or 10
        self.device = kwargs.get("device") or CONFIGS.torch.device

    def run(self, savepath: str, name = None, ):        
        with Format(Format.BOLD, Format.YELLOW):
            print(f"Running experiment {self.experiment_name}")
            
        run_name = name or namegenerator.gen()

        if hasattr(self, "ratio"):
            with Format(Format.BOLD, Format.BRIGHT_MAGENTA):
                print(f"Running run {run_name}")
                
            self.run_savepath_ = os.path.join(savepath, self.experiment_name, run_name)
            if not os.path.exists(self.run_savepath_):
                os.makedirs(self.run_savepath_)
                
            self._train(self.ratio)
            self._test()
        else:
            for runidx, ratio in enumerate(np.linspace(self.rstart, self.rstop, self.rnum), 1):
                with Format(Format.BOLD, Format.BRIGHT_MAGENTA):
                    print(f"Running run {run_name}-{runidx}")
                    
                self.run_savepath_ = os.path.join(savepath, self.experiment_name, run_name, str(runidx))
                if not os.path.exists(self.run_savepath_):
                    os.makedirs(self.run_savepath_)
                
                self._train(ratio)
                self._test()

    def _train(self, ratio: float):
        sampler_tr = self.sampler_cls(self.dataset_tr_, ratio)
        
        num_train = int(0.8 * len(sampler_tr))
        num_valid = len(sampler_tr) - num_train
        
        train_dataset, valid_dataset = random_split(sampler_tr, [num_train, num_valid])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model_.parameters(), lr=0.001)
        best_valid_acc = 0
        
        self.model_.train()
        
        for epoch in range(self.epochs):            
            self.__train_epoch(train_loader, epoch, criterion, optimizer)
            best_valid_acc = self.__valid_epoch(valid_loader, best_valid_acc)
        
    def __train_epoch(self, train_loader, epoch, criterion, optimizer):
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit=" batch") as pbar:
            for batchno, (inputs, labels) in enumerate(train_loader, start=1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{running_loss / batchno:.4f}")
    
    def __valid_epoch(self, valid_loader, best_valid_acc):
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc="Validating", unit=" batch") as pbar:
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model_(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    pbar.update(1)
                
        accuracy = correct / total
        
        if accuracy > best_valid_acc:
            with Format(Format.BOLD, Format.CYAN):
                print(f"Best valid accuracy improved from {best_valid_acc * 100:.2f}% to {accuracy * 100:.2f}%. Saving checkpoint...")
            best_valid_acc = accuracy
            torch.save(self.model_.state_dict(), f"{self.run_savepath_}/model.pth")
            
        return best_valid_acc

    def _test(self):
        test_loader = DataLoader(self.dataset_te_, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        self.model_.load_state_dict(torch.load(f"{self.run_savepath_}/model.pth"))

        self.model_.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Testing", unit=" batch") as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model_(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    pbar.update(1)

        accuracy = correct / total
        
        with open(os.path.join(self.run_savepath_, 'acc.txt'), 'w+') as f:
            f.write(f"{100 * accuracy:.2f}%")


