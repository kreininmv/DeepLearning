# Torch import
import torch
import math
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn import BatchNorm3d
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinarySpecificity
from torch.cuda.amp import GradScaler
# Stuff
import pickle
import gc
from sklearn.model_selection import train_test_split
import os, glob
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
# Monai import
from monai.inferers import sliding_window_inference
from functools import partial
import monai
from monai.metrics import DiceMetric, DiceHelper
from monai.networks import one_hot
from monai.visualize import plot_2d_or_3d_image
import random
import segmentation_models_pytorch as smp
from itertools import chain
from torchvision.transforms import ToTensor, PILToTensor, Compose
from monai.transforms import ToTensord
import albumentations as A
import typing as tp
from PIL import Image
import matplotlib.pyplot as plt
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
import torchvision
from monai.transforms import (Compose, 
                            DivisiblePadd, 
                            AddChannelD, 
                            CenterSpatialCropd, 
                            RandFlipd,
                            RandSpatialCropd,
                            ScaleIntensityRanged)


def crop_or_pad_tensor(
        tensor: torch.Tensor,
        target_shape: tp.Sequence[int],
) -> tp.Tuple[torch.Tensor, tp.List[tp.List[int]], tp.List[tp.List[int]]]:
    """
    Crops (if target shape is less than current) or pads tensor.
    Ignores first dimensions if len(target_shape) is less than len(tensor.shape)

    Function returns cropped region and padding in format
    [[first_dim_0, last_dim_0], ...], [[pad_before_dim_0, pad_after_dim_0], ...]
    """
    start_dim = len(tensor.shape) - len(target_shape)
    if start_dim < 0:
        raise ValueError(f'Target tensor has more dimensions ({len(target_shape)})'
                         f' than source tensor ({len(tensor.shape)})')

    rois: tp.List[tp.List[int]] = []
    pads: tp.List[tp.List[int]] = []
    flattened_pads = []
    for i in range(start_dim, len(tensor.shape)):
        roi, pad = normalize_roi([0, tensor.shape[i] - 1], int(target_shape[i - start_dim]), tensor.shape[i])
        tensor = tensor.transpose(0, i)[roi[0]:roi[1] + 1].transpose(0, i)
        flattened_pads.extend(pad[::-1])
        rois.append(roi)
        pads.append(pad)

    tensor = F.pad(
        tensor,
        tuple(flattened_pads[::-1]),
        'constant',
        0
    )

    return tensor, rois, pads


def divisible_pad(
        tensor: torch.Tensor,
        shape_divider: int,
        start_dim: int = 0,
) -> tp.Tuple[torch.Tensor, tp.List[tp.List[int]], tp.List[tp.List[int]]]:
    """
    Divisible padding of tensor

    First start_dim dimensions are not padded

    Return format is the same to crop_or_pad_tensor
    """
    new_shape = (np.ceil(np.array(tensor.shape[start_dim:]) / shape_divider) * shape_divider).astype(int).tolist()
    return crop_or_pad_tensor(tensor, new_shape)

def normalize_roi(
        roi: tp.Sequence[int],
        desired_length: int,
        shape: int,
) -> tp.Tuple[tp.List[int], tp.List[int]]:
    """
    Extends the ROI to desired length

    Return new ROI and padding, that should be applied to cropped array
    """
    pad = [0, 0]

    roi = [
        (roi[0] + roi[1]) // 2 - desired_length // 2 + 1 - desired_length % 2,
        (roi[0] + roi[1]) // 2 + desired_length // 2
    ]
    if roi[0] < 0:
        pad[0] = -roi[0]
        roi[0] = 0
    if roi[1] > shape - 1:
        pad[1] = roi[1] - shape + 1
        roi[1] = shape - 1
    return roi, pad



class FloodNet(Dataset):
    """
    Классы:
    0: Background, 1: Building, 2: Road, 3: Water, 4: Tree, 5: Vehicle, 6: Pool, 7: Grass
    """
    def __init__(
        self,
        data_path: str,
        phase: str,
        augment: bool,
        img_size: int,
    ):
        self.num_classes = 8
        self.data_path = data_path
        self.phase = phase
        self.augment = augment
        self.img_size = img_size
        self.to_tensor = ToTensor()
        
        #if ph#ase != 'inference' and phase != 'danya':
            #self.items = [filename.split('.')[0] for filename in os.listdir(f'{data_path}/{phase}/image')]

        if phase == 'train':
            self.items = [filename.split('.')[0] for filename in os.listdir(f'{data_path}/{phase}/image')]
            self.transform = Compose([AddChannelD(keys=['mask']),
                                      DivisiblePadd(keys=['image', 'mask'], k=32, method='symmetric', mode='constant'),
                                      CenterSpatialCropd(keys=['image', 'mask'], roi_size=[768, 1024]),
                                      RandFlipd(keys=['image', 'mask'], spatial_axis=0, prob=0.5),
                                      RandFlipd(keys=['image', 'mask'], spatial_axis=1, prob=0.5),
                                      ToTensord(keys=['image', 'mask'])])    
        elif phase == 'test':
            self.items = [filename.split('.')[0] for filename in os.listdir(f'{data_path}/{phase}/image')]
            self.transform = Compose([AddChannelD(keys=['mask']),
                                      DivisiblePadd(keys=['image', 'mask'], k=32, method='symmetric', mode='constant'),
                                      ToTensord(keys=['image', 'mask'])]) 
        else:
            self.items = [filename.split('.')[0] for filename in os.listdir(f'{data_path}/test/image')]
            self.phase = 'test'
            self.transform = Compose([
                AddChannelD(keys=['mask']), 
                ScaleIntensityRanged(keys=['image'], a_min=0, a_max=250, b_min=0.0, b_max=1.0, clip=True), 
                DivisiblePadd(keys=['image', 'mask'], k=32, method='symmetric', mode='constant'),
                ToTensord(keys=['image', 'mask'])
                ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        
        
        image = np.asarray(Image.open(f'{self.data_path}/{self.phase}/image/{self.items[index]}.jpg'))
        mask = np.asarray(Image.open(f'{self.data_path}/{self.phase}/mask/{self.items[index]}.png'))
        
        image = self.to_tensor(image.copy())
        mask = torch.from_numpy(mask.copy()).long()
        
        
        tmp = self.transform({'image': image, 'mask': mask})
        
        return tmp['image'], tmp['mask'][0]
    
def dice_loss(net_output, target, class_label=torch.tensor(1).type(torch.int), eps=1):
    target_ = (target == class_label)
    return 1 - (2 * torch.sum(net_output * target_) + 1) / (torch.sum(net_output) + torch.sum(target_) + eps)

def iou(net_output, target, class_label=torch.tensor(1).type(torch.int), eps=1):
    net_out_ = (net_output == class_label)
    target_ = (target == class_label)
    return (torch.sum(net_out_ * target_))/(torch.sum(torch.maximum(net_out_, target_)) + eps)

def recall(net_output, target, class_label=torch.tensor(1).type(torch.int), eps=1):
    net_out_ = (net_output == class_label)
    target_ = (target == class_label)
    return (torch.sum(net_out_ * target_))/(torch.sum(target_) + eps)

def accuracy(net_output, target, class_label=torch.tensor(1).type(torch.int), eps=1):
    net_out_ = (net_output == class_label)
    target_ = (target == class_label)
    return (torch.sum(net_out_ == target_))/(torch.sum(torch.ones_like(target_))+ eps)
    
def tversky_loss(outputs, targets, alpha=0.5, gamma=0.5):
    smooth = 1
    tversky_loss = 0
    beta = 1 - alpha
    output = outputs.reshape(-1)
    target = targets.reshape(-1)
    intersection = (output*target).sum()
    FP = ((1 - target)*output).sum()
    FN = (target*(1 - output)).sum()
    tv = (intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)
    tversky = 1. - tv
    return tversky**gamma

class Trainer:        
    def set_wandb(self, wandb_init):
        self.config = {
                'architecture'    : 'BasiUNet++',
                'spatial_dims'    : 2,
                'in_channels'     : 3,
                'out_channels'    : 8, 
                'encoder_name'    : 'resnet152', 
                'encoder_weights' : 'imagenet',
                'activation'      : None,
                'optimizer'       : 'AdamW',
                'learning_rate'   : 1e-3,
                'weight_decay'    : 1e-5,
                'optimizer_kwargs': {'betas': (0.9, 0.999), 'eps': 1e-7},
                'scheduler_name'  : 'StepLR',
                'scheduler_kwargs': {'eta_min': 2e-4, 'T_max': 300, 'gamma': 0.95, 'step_size': 7, },
                'epochs'          : 300,
                'batch_size'      : 7
            }
        torch.cuda.set_device(3)
        self.wandb = wandb_init

        if self.wandb:
            wandb.login(key='84d6a92704bf4bf19d2ecc87a55eea5ce77a8725')
            self.run = wandb.init(project='hw4', config=self.config)
            self.model_name = 'run_' + self.run.name + '_model'
        else:
            self.model_name = 'testing'
            
        self.start_epoch = 0
        self.flag = 0
        self.num_epochs  = self.config['epochs']
        
    def __init__(self, wandb_init=False):
        # Get  files
        self.set_wandb(wandb_init)
        self.set_net()
        self.set_opt_sched()
        self.set_loaders()
        # Initialize all essentials critetia and metrics
        weights = torch.tensor([1., 1., 1., 1., 1., 4., 4., 2.]).to(self.device).float()
        self.entropy  = nn.CrossEntropyLoss(weight=weights)
        self.SoftMax  = torch.nn.Softmax(dim=1)
        self.names = {
                    0: 'background',
                    1: 'Building',     
                    2: 'Road',
                    3: 'Water',
                    4: 'Tree',
                    5: 'Vehicle',
                    6: 'Pool',
                    7: 'Grass'
        }
        self.focal    = monai.losses.FocalLoss(include_background=False, 
                                               to_onehot_y=True, 
                                               gamma=2, 
                                               alpha=0.4,
                                               weight=torch.tensor([1., 1., 1., 1., 4., 4., 1.], device=self.device).float(),
                                               use_softmax=True)
        #self.padder = DivisiblePad(k=32, method='symmetric', mode='constant')
        self.best_acc = 0
        self.transform = A.Resize(width=1000, height=750)
        self.to_tensor = ToTensor()
    
    def criterion(self, outputs, targets, config):
        probability = torch.softmax(outputs, dim=1)

        loss_dice =     dice_loss(probability[:, 0], targets, 0) +\
                        dice_loss(probability[:, 1], targets, 1) +\
                        dice_loss(probability[:, 2], targets, 2) +\
                        dice_loss(probability[:, 3], targets, 3) +\
                        dice_loss(probability[:, 4], targets, 4) +\
                      2*dice_loss(probability[:, 5], targets, 5) +\
                      2*dice_loss(probability[:, 6], targets, 6) +\
                        dice_loss(probability[:, 7], targets, 7)
        
            
        loss_entropy = self.entropy(probability, targets)
        
        loss_focal   = self.focal(outputs, targets.unsqueeze(1))
        loss = loss_dice + 2*loss_entropy + 2*loss_focal
        
        config['loss'].append(loss.item())
        config['entropy_loss'].append(loss_entropy.item())
        config['dice_loss'].append(loss_dice.item())
        config['focal_loss'].append(loss_focal.item())
            
        return loss
    
    def metrics(self, outputs, targets, config):
        probability = torch.softmax(outputs, dim=1)
        predict = torch.argmax(probability, dim=1)
        mAcc, mIoU, mRecall = [], [], []
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = iou(predict, targets, i).item()
                config[f'IoU [{self.names[i]}]'].append(tmp_)
                mIoU.append(tmp_)
        
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = recall(predict, targets, i).item()
                config[f'Recall [{self.names[i]}]'].append(tmp_)
                mRecall.append(tmp_)
            
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = accuracy(predict, targets, i).item()
                config[f'Accuracy [{self.names[i]}]'].append(tmp_)
                mAcc.append(tmp_)
        
        config['IoU'].append(np.mean(mIoU))
        config['Recall'].append(np.mean(mRecall))
        config['Accuracy'].append(np.mean(mAcc))
    
    def danya_metrics(self, predict, targets, config):
        mAcc, mIoU, mRecall = [], [], []
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = iou(predict, targets, i).item()
                config[f'IoU [{self.names[i]}]'].append(tmp_)
                mIoU.append(tmp_)
        
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = recall(predict, targets, i).item()
                config[f'Recall [{self.names[i]}]'].append(tmp_)
                mRecall.append(tmp_)
            
        for i in range(len(self.names)):
            if i in targets:
                tmp_ = accuracy(predict, targets, i).item()
                config[f'Accuracy [{self.names[i]}]'].append(tmp_)
                mAcc.append(tmp_)
        
        config['IoU'].append(np.mean(mIoU))
        config['Recall'].append(np.mean(mRecall))
        config['Accuracy'].append(np.mean(mAcc))
    def wandb_log(self, config, name):    
        wandb.log({f'{name} dice loss'            : np.mean(config['dice_loss']), 
                   f'{name} focal loss'           : np.mean(config['focal_loss']),
                   f'{name} loss'                 : np.mean(config['loss']),
                   f'{name} entropy_loss'         : np.mean(config['entropy_loss']),
                   f'{name} LR'                   : float(get_lr(self.optimizer)),
                   'Epoch'                        : config['i_epoch'],
                   f'{name} IoU'                  : np.mean(config['IoU']),
                   f'{name} Recall'               : np.mean(config['Recall']),
                   f'{name} Accuracy'             : np.mean(config['Accuracy']),
                   f'{name} IoU [background]'     : np.mean(config['IoU [background]']),
                   f'{name} IoU [Building]'       : np.mean(config['IoU [Building]']),
                   f'{name} IoU [Road]'           : np.mean(config['IoU [Road]']),
                   f'{name} IoU [Water]'          : np.mean(config['IoU [Water]']),
                   f'{name} IoU [Tree]'           : np.mean(config['IoU [Tree]']),
                   f'{name} IoU [Vehicle]'        : np.mean(config['IoU [Vehicle]']),
                   f'{name} IoU [Pool]'           : np.mean(config['IoU [Pool]']),
                   f'{name} IoU [Grass]'          : np.mean(config['IoU [Grass]']),
                   f'{name} Recall [background]'  : np.mean(config['Recall [background]']),
                   f'{name} Recall [Building]'    : np.mean(config['Recall [Building]']),
                   f'{name} Recall [Road]'        : np.mean(config['Recall [Road]']),
                   f'{name} Recall [Water]'       : np.mean(config['Recall [Water]']),
                   f'{name} Recall [Tree]'        : np.mean(config['Recall [Tree]']),
                   f'{name} Recall [Vehicle]'     : np.mean(config['Recall [Vehicle]']),
                   f'{name} Recall [Pool]'        : np.mean(config['Recall [Pool]']),
                   f'{name} Recall [Grass]'       : np.mean(config['Recall [Grass]']),
                   f'{name} Accuracy [background]': np.mean(config['Accuracy [background]']),
                   f'{name} Accuracy [Building]'  : np.mean(config['Accuracy [Building]']),
                   f'{name} Accuracy [Road]'      : np.mean(config['Accuracy [Road]']),
                   f'{name} Accuracy [Water]'     : np.mean(config['Accuracy [Water]']),
                   f'{name} Accuracy [Tree]'      : np.mean(config['Accuracy [Tree]']),
                   f'{name} Accuracy [Vehicle]'   : np.mean(config['Accuracy [Vehicle]']),
                   f'{name} Accuracy [Pool]'      : np.mean(config['Accuracy [Pool]']),
                   f'{name} Accuracy [Grass]'     : np.mean(config['Accuracy [Grass]']),
                  })
    
    def test(self, i_epoch):
        # Test
        self.net.eval()
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        
        with torch.no_grad():
            loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
            for batch_idx, (inputs, targets) in loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)   
                
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, config)
                # Calculate and summary metrics
                self.metrics(outputs, targets, config)
                # LOOPA and PUPA
                loop.set_description(f"Epoch (Test)[{i_epoch}/{self.num_epochs}]")
                loop.set_postfix(IoU=np.mean(config['IoU']), loss=np.mean(config['loss']))
                gc.collect()
                torch.cuda.empty_cache()
                
            config['i_epoch'] = i_epoch
            self.wandb_log(config, name='Test')

        # Save checkpoint.
        acc = 100.*np.mean(config['IoU'])
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(i_epoch, self.model_name + f'_best')
    
    def train(self, i_epoch):
        # Train
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        self.net.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets, config)
            gc.collect()
            torch.cuda.empty_cache()
            # Make backward step
            loss.backward()
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            # Calculate and summary metrics
            self.metrics(outputs, targets, config)
            # LOOPA and PUPA
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(IoU=np.mean(config['IoU']), loss=np.mean(config['loss']))
            
        gc.collect()
        torch.cuda.empty_cache()
        config['i_epoch'] = i_epoch
        self.wandb_log(config, name='Train')
            
    def fit(self):        
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.test(i_epoch)        
            self.scheduler.step()
            self.set_loaders()
            if self.flag:
                break
        self.run.finish()
        
    def set_opt_sched(self):
        self.optimizer = optim.AdamW(
            params       = self.net.parameters(),
            lr           = self.config['learning_rate'],
            betas        = self.config['optimizer_kwargs']['betas'],
            eps          = self.config['optimizer_kwargs']['eps'],
            weight_decay = self.config['weight_decay']
            )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['scheduler_kwargs']['step_size'],
            gamma=self.config['scheduler_kwargs']['gamma']
            )
        
        
            
    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.net = smp.Unet(encoder_name   = self.config['encoder_name'], 
                              encoder_weights= self.config['encoder_weights'],
                              activation     = self.config['activation'],
                              in_channels    = self.config['in_channels'],
                              classes        = self.config['out_channels']
                             )
        
        #self.net = PSPNet(n_classes=8,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
        self.net = self.net.to(self.device)
        
        if self.wandb:
            wandb.watch(self.net, log_freq=100)
     
    def load(self, filename):
        checkpoint = torch.load(f'./checkpoints/{filename}.pth', map_location=torch.device('cpu'))
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net         = self.net.to(self.device)
        self.best_acc    = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']        
        if self.wandb:
            wandb.watch(self.net, log_freq=100)
        
        self.second_net =  torchvision.models.segmentation.deeplabv3_resnet101(num_classes=8,
                                                                               weights_backbone=torchvision.models.ResNet101_Weights.IMAGENET1K_V2) 
 
        # Создание нового классификационного слоя 
        num_classes = 8 
        self.second_net.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
 
        # Загружаем веса из файла best_model.pth 
        self.second_net.load_state_dict(torch.load("./danya.pth", map_location=torch.device('cpu'))) 
 
        self.second_net = self.second_net.to(self.device) 
        self.second_net.eval()
        
    def save_model(self, i_epoch, name):
        state = {
            'net'      : self.net.state_dict(), 
            'acc'      : self.best_acc, 
            'epoch'    : i_epoch, 
            }
        torch.save(state, f'./checkpoints/{name}.pth')
        print(f'Saving... Epoch[{i_epoch}]')
    
    def set_loaders(self):
        train_dataset = FloodNet(data_path='tiny-floodnet-challenge',phase='train', augment=False, img_size=256)
        self.train_loader = DataLoader(train_dataset, num_workers=1, batch_size=self.config['batch_size'], shuffle=True)
        
        test_dataset = FloodNet(data_path='tiny-floodnet-challenge', phase='test', augment=False, img_size=256)
        self.test_loader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False)

    def visualise(self, img_, out_, mask_, pad = None, config=None):
        out = torch.softmax(out_, dim=0)
        out = torch.argmax(out, dim=0)
        
        pad_x, pad_y = pad
        
        img = img_[:, pad_x[0]:mask_.shape[0] - pad_x[1], pad_y[0]:mask_.shape[1] - pad_y[1]].numpy()
        out = out[pad_x[0]:mask_.shape[0] - pad_x[1], pad_y[0]:mask_.shape[1] - pad_y[1]].unsqueeze(-1).numpy()
        mask = mask_[pad_x[0]:mask_.shape[0] - pad_x[1], pad_y[0]:mask_.shape[1] - pad_y[1]].unsqueeze(-1).numpy()
        
       
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(36, 10))
        
        im2display = img.transpose((1,2,0))
        im2display *= 255
        ax[0].imshow(im2display.astype(float).clip(0, 1), interpolation='nearest')
        #ax[0].imshow(im2display, interpolation='nearest')
        ax[0].set_title('Original image', fontsize = '24')
        ax[0].axis('off')
        
        ax[1].imshow(mask.astype(int), interpolation='nearest', cmap='plasma')
        ax[1].set_title('Original mask', fontsize = '24')
        ax[1].axis('off')
        
        ax[2].imshow(out.astype(int), interpolation='nearest', cmap='plasma')
        ax[2].set_title('My Prediction', fontsize = '24')
        ax[2].axis('off')
        
        
        plt.show()
    
        
    def inference(self, data_path='tiny-floodnet-challenge'):
        inference_dataset = FloodNet(data_path='tiny-floodnet-challenge', phase='inference', augment=False, img_size=256)
        self.val_loader = DataLoader(inference_dataset, num_workers=1, batch_size=1, shuffle=False)
        
        self.net.eval()
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        
        with torch.no_grad():
            loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False, disable=True)
            for batch_idx, (inputs_orig, targets_orig) in loop:
                inputs_orig, targets_orig = inputs_orig.to(self.device), targets_orig.to(self.device)   
                inputs, _, pad = divisible_pad(inputs_orig, 32, 2)
                targets, _, pad = divisible_pad(targets_orig, 32, 1)
                
                outputs = self.net(inputs)
                
                loss = self.criterion(outputs, targets, config)
                # Calculate and summary metrics
                self.metrics(outputs, targets, config)
                # LOOPA and PUPA
                
                loop.set_description(f"Epoch (Inference)")
                loop.set_postfix(IoU=np.mean(config['IoU']), loss=np.mean(config['loss']))
                gc.collect()
                torch.cuda.empty_cache()
                self.visualise(inputs[0].cpu(), outputs[0].cpu(), targets[0].cpu(), pad)
                
                
        return config
    
    #0: Background, 1: Building, 2: Road, 3: Water, 4: Tree, 5: Vehicle, 6: Pool, 7: Grass : 8
    def get_config(self):
        return {
            'loss'                 : [],
            'entropy_loss'         : [],
            'focal_loss'           : [],
            'dice_loss'            : [],
            'i_epoch'              : 0,
            'IoU'                  : [],
            'Recall'               : [],
            'Accuracy'             : [],
            'IoU [background]'     : [],
            'IoU [Building]'       : [],
            'IoU [Road]'           : [],
            'IoU [Water]'          : [],
            'IoU [Tree]'           : [],
            'IoU [Vehicle]'        : [],
            'IoU [Pool]'           : [],
            'IoU [Grass]'          : [],
            'Recall [background]'  : [],
            'Recall [Building]'    : [],
            'Recall [Road]'        : [],
            'Recall [Water]'       : [],
            'Recall [Tree]'        : [],
            'Recall [Vehicle]'     : [],
            'Recall [Pool]'        : [],
            'Recall [Grass]'       : [],
            'Accuracy [background]': [],
            'Accuracy [Building]'  : [],
            'Accuracy [Road]'      : [],
            'Accuracy [Water]'     : [],
            'Accuracy [Tree]'      : [],
            'Accuracy [Vehicle]'   : [],
            'Accuracy [Pool]'      : [],
            'Accuracy [Grass]'     : [],
            }