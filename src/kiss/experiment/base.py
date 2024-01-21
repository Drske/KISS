import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import namegenerator

from tqdm import tqdm
from typing import Tuple
from copy import deepcopy
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import random_split

from kiss.sampler._sampler import Sampler
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

        self.model = model
        self.model_ = deepcopy(model)
        self.dataset_tr_ = dataset_tr
        self.dataset_te_ = dataset_te

        if isinstance(ratio, (float, int)):
            self.ratio = ratio

        if isinstance(ratio, tuple):
            self.rstart = ratio[0]
            self.rstop = ratio[1]
            self.rnum = ratio[2]

        self.sampler_ : Sampler = sampler_cls(dataset_tr, ratio, **kwargs)
        
        self.experiment_name = name or f"{self.model.__class__.__name__}!{dataset_tr.__class__.__name__}!{self.sampler_.__class__.__name__}"
        
        self.batch_size = kwargs.get("batch_size", 512)
        self.num_workers = kwargs.get("num_workers", 0)
        self.epochs = kwargs.get("epochs", 10)
        self.clip = kwargs.get("clip", None)
        self.device = kwargs.get("device", CONFIGS.torch.device)

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
                    print(f"Running run {run_name}/{runidx}")
                    
                self.run_savepath_ = os.path.join(savepath, self.experiment_name, run_name, str(runidx))
                if not os.path.exists(self.run_savepath_):
                    os.makedirs(self.run_savepath_)
                
                self.model = deepcopy(self.model_)
                self._train(ratio)
                self._test()

    def _train(self, ratio: float):
        self.sampler_.calculate_indices(ratio)
        
        num_train = int(0.8 * len(self.sampler_))
        num_valid = len(self.sampler_) - num_train
        
        train_dataset, valid_dataset = random_split(self.sampler_, [num_train, num_valid])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters())
        
        best_valid_loss = np.inf
        best_valid_acc = 0.0
        
        self.model.train()
        
        for epoch in range(self.epochs):            
            self.__train_epoch(train_loader, epoch, criterion, optimizer)
            best_valid_loss, best_valid_acc = self.__valid_epoch(valid_loader, criterion, best_valid_loss, best_valid_acc)
            
        with open(os.path.join(self.run_savepath_, 'ratio.txt'), 'w+') as f:
            f.write(f"{ratio:.2f}")
            
        with open(os.path.join(self.run_savepath_, 'train_size.txt'), 'w+') as f:
            f.write(f"{num_train}")
            
        with open(os.path.join(self.run_savepath_, 'valid_size.txt'), 'w+') as f:
            f.write(f"{num_valid}")
        
    def __train_epoch(self, train_loader, epoch, criterion, optimizer):
        self.model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit=" batch") as pbar:
            for batchno, (inputs, labels) in enumerate(train_loader, start=1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{running_loss / batchno:.4f}")
    
    def __valid_epoch(self, valid_loader, criterion, best_valid_loss, best_valid_acc):
        self.model.eval()
        
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc="Validating", unit=" batch") as pbar:
                for batchno, (inputs, labels) in enumerate(valid_loader, start=1):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    running_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{running_loss / batchno:.4f}")
                
        accuracy = correct / total
        valid_loss = running_loss / len(valid_loader)
        
        if valid_loss < best_valid_loss:
            with Format(Format.BOLD, Format.CYAN):
                print(f"Best valid loss improved. Current accuracy is {accuracy * 100:.2f}%. Saving checkpoint...")
            best_valid_loss = valid_loss
            torch.save(self.model.state_dict(), f"{self.run_savepath_}/model.pth")
            
        if accuracy > best_valid_acc:
            with Format(Format.BOLD, Format.CYAN):
                print(f"Best valid accuracy improved. Current accuracy is {accuracy * 100:.2f}%. Saving checkpoint...")
            best_valid_acc = accuracy
            torch.save(self.model.state_dict(), f"{self.run_savepath_}/model.pth")
            
        return best_valid_loss, best_valid_acc

    def _test(self):
        test_loader = DataLoader(self.dataset_te_, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        self.model.load_state_dict(torch.load(f"{self.run_savepath_}/model.pth"))

        self.model.eval()
        top1 = 0
        top3 = 0
        top5 = 0
        total = 0

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Testing", unit=" batch") as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted_top1 = torch.max(outputs, 1)
                    _, predicted_top3 = torch.topk(outputs, 3)
                    _, predicted_top5 = torch.topk(outputs, 5)
                    
                    
                    top1 += (predicted_top1 == labels).sum().item()
                    top3 += torch.sum(torch.any(predicted_top3 == labels.view(-1, 1), dim=1)).item()
                    top5 += torch.sum(torch.any(predicted_top5 == labels.view(-1, 1), dim=1)).item()
                    total += labels.size(0)

                    pbar.update(1)

        acc_top1 = top1 / total
        acc_top3 = top3 / total
        acc_top5 = top5 / total
        
        with open(os.path.join(self.run_savepath_, 'acc.txt'), 'w+') as f:
                f.write(f"{100 * acc_top1:.2f}%")
                
        with open(os.path.join(self.run_savepath_, 'top3.txt'), 'w+') as f:
                f.write(f"{100 * acc_top3:.2f}%")
                
        with open(os.path.join(self.run_savepath_, 'top5.txt'), 'w+') as f:
                f.write(f"{100 * acc_top5:.2f}%")
            
        with open(os.path.join(self.run_savepath_, 'test_size.txt'), 'w+') as f:
            f.write(f"{len(self.dataset_te_)}")



