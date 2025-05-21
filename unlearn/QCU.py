import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from time import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from quantization_8bit import QuantizedResNet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlearningModel:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = DEVICE
        self.quantized_model = QuantizedResNet18(deepcopy(model)).to(self.device)
        self.distances = {'positive': [], 'negative': []}

    def _create_optimizers(self):
        return {
            'forget': torch.optim.Adam(self.model.parameters(), 0.0014, weight_decay=0.001),
            'retain': torch.optim.Adam(self.model.parameters(), 0.0003, weight_decay=0.003)
        }

    def unlearn(self, data_loaders, args):
        optimizers = self._create_optimizers()
        forget_steps = len(data_loaders["forget"]) * 2
        retain_steps = len(data_loaders["retain"]) * 8
        schedulers = {
            'forget': CosineAnnealingLR(optimizers['forget'], T_max=forget_steps),
            'retain': CosineAnnealingLR(optimizers['retain'], T_max=retain_steps)
        }

        start = time()
        original_model = deepcopy(self.model).to(self.device).eval()
        self.model.train()

        for _ in range(3):
            self.QuantizedContrastiveUnlearning(
                data_loaders["forget"], 
                original_model, 
                optimizers['forget'], 
                schedulers['forget']
            )
        
        for _ in range(7):
            self.FT(
                data_loaders["retain"], 
                optimizers['retain'], 
                schedulers['retain']
            )

        print(f'Elapsed time written by minutes: {(time()-start)/60}')
        return self.model

    def _create_subset_loader(self, loader, ratio=0.1):
        num_samples = int(len(loader.dataset) * ratio)
        subset_loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(range(num_samples)),
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        return subset_loader

    def QuantizedContrastiveUnlearning(self, train_loader, original_model, optimizer, scheduler):
        for image, _ in tqdm(train_loader):
            image = image.to(self.device)
            
            with torch.no_grad():
                negative = self._extract_logit(original_model, image)
                positive = self._extract_logit(self.quantized_model, image)
            anchor = self._extract_logit(self.model, image)
            
            loss = self.contrastiveLoss(
                anchor=anchor,
                positive=positive,
                negative=negative,
                margin=1.0,
                alpha=1.0,
                beta=0.9
            )
            
            self.distances['negative'].append(
                torch.sum((anchor - negative) ** 2, dim=1).mean().item()
            )
            self.distances['positive'].append(
                torch.sum((anchor - positive) ** 2, dim=1).mean().item()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    def _extract_logit(self, model, image):
        features = torch.nn.Sequential(*list(model.children())[:-1])
        features = features.to(self.device)
        x = features(image)
        x = torch.flatten(x, 1)
        logit = model.fc(x)
        return logit

    def FT(self, retain_loader, optimizer, scheduler):
        losses = []
        for image, target in tqdm(retain_loader):
            image, target = image.to(self.device), target.to(self.device)
            
            loss = F.cross_entropy(self.model(image), target)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        print(f'Loss: {np.mean(losses)}')

    def contrastiveLoss(self, anchor, positive, negative, margin=1.0, alpha=0.5, beta=0.8):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        weighted_loss = alpha * dist_pos - beta * dist_neg + margin
        loss = torch.clamp(weighted_loss, min=0.0)
        return loss.mean()


def QCU(data_loader, model, criterion, args, mask=None):
    unlearning_model = UnlearningModel(model, args)
    return unlearning_model.unlearn(data_loader, args)