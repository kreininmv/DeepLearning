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
from monai.losses import DiceLoss
from monai.transforms import (
    AddChannelD,    
    Compose,
    CenterSpatialCropd,  
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    RandShiftIntensityd,
    RandAffined,
    RandSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    ResizeWithPadOrCropd,    
    RandFlipd,
    RandZoomd,
    RandScaleIntensityd,
    SpatialPadd,
    ScaleIntensityRanged,
    MedianSmoothd,
    ToTensord,
)
import random
from random import choice
import h5py
import json
import math
import segmentation_models_pytorch as smp
from itertools import chain
from collections import Counter

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class IdxDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        return self.dataset[self.idx[idx]]

class ScaledDotProductScore(nn.Module):
    """
    Vaswani et al. "Attention Is All You Need", 2017.
    """
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):
        """
        queries:  [batch_size x num_queries x dim]
        keys:     [batch_size x num_objects x dim]
        Returns a tensor of scores with shape [batch_size x num_queries x num_objects].
        """
        return torch.bmm(queries, keys.permute(0, 2, 1))/math.sqrt(keys.shape[-1])

class Attention(nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, queries, keys, values):
        """
        queries:         [batch_size x num_queries x query_feature_dim]
        keys:            [batch_size x num_objects x key_feature_dim]
        values:          [batch_size x num_objects x obj_feature_dim]
        Returns matrix of responses for queries with shape [batch_size x num_queries x obj_feature_dim].
        Saves detached weights as self.attention_map.
        """
        if len(queries.shape) == 2:
            queries = queries.unsqueeze(1)
            
        scores = self.scorer(queries, keys)
        weights = torch.exp(scores - torch.log(torch.sum(torch.exp(scores), dim=2, keepdims=True)))
        self.attention_map = weights.detach()
        
        return torch.bmm(weights, values)

class CaptionNet(nn.Module):
    def __init__(self, n_tokens=10403, emb_size=128, lstm_units=256, cnn_channels=512):
        """ A recurrent 'head' network for image captioning. Read scheme below. """
        super(self.__class__, self).__init__()
        self.cnn_to_h0 = nn.Linear(in_features=cnn_channels, out_features=lstm_units, bias=True)
        self.cnn_to_c0 = nn.Linear(in_features=cnn_channels, out_features=lstm_units, bias=True)
        self.emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size)
        self.attention = Attention(ScaledDotProductScore())
        self.cnn_to_attn_key = nn.Linear(in_features=cnn_channels, out_features=lstm_units, bias=True)
        self.lstm = torch.nn.LSTMCell(input_size=emb_size+cnn_channels, hidden_size=lstm_units, bias=True)        
        self.logits_mlp = nn.Linear(in_features=lstm_units+cnn_channels+emb_size, out_features=n_tokens)

    def forward(self, image_features, captions_ix):
        """
        Apply the network in training mode.
        :param image_features: torch tensor containing VGG features for each position.
                               shape: [batch, cnn_channels, width * height]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """
        initial_cell = self.cnn_to_c0(image_features.mean(2))
        initial_hid = self.cnn_to_h0(image_features.mean(2))
        image_features = image_features.transpose(1, 2)
        keys = self.cnn_to_attn_key(image_features)
        captions_emb = self.emb(captions_ix)

        ## Предварительные танцы с бубном (первый шаг)
        attention = self.attention.forward(queries=initial_cell, keys=keys, values=image_features) # берем первый аттеншен 
        attentions_maps = self.attention.attention_map # запоминаем первый аттеншен map
        input = torch.concat([captions_emb[:, 0], attention.squeeze(1)], dim=1) # соединили эмбеддинг и attention map
        h, c = self.lstm(input, (initial_hid, initial_cell)) # lstm сделала шаг
        logits = self.logits_mlp(torch.concatenate([attention, captions_emb[:, 0].unsqueeze(1), h.unsqueeze(1)], dim=2))
        # повторяем всё тоже самое
        for i in range(1, captions_ix.shape[1]):
            attention = self.attention(queries=h, keys=keys, values=image_features) 
            attentions_maps = torch.concat([attentions_maps, self.attention.attention_map], dim=1)
            input = torch.concat([captions_emb[:, i], attention.squeeze(1)], dim=1)
            h, c = self.lstm(input, (h, c))
            tmp_logit = self.logits_mlp(torch.concatenate([attention, captions_emb[:, i].unsqueeze(1), h.unsqueeze(1)], dim=2))
            logits = torch.concat([logits, tmp_logit], dim=1)
        
        return logits, attentions_maps

class Trainer:        
    def set_wandb(self, wandb_init):
        #wandb init изменить
        self.wandb_init = wandb_init
        self.config = {
                'architecture'    : 'CaptionNet',
                'n_tokens'        : self.n_tokens, 
                'emb_size'        : 2048, 
                'lstm_units'      : 2048,
                'cnn_channels'    : 512,            
                'optimizer'       : 'SGD',
                'learning_rate'   : 5e-3,
                'weight_decay'    : 1e-6,
                'momentum'        : 0.95,
                'optimizer_kwargs': {'betas': (0.9, 0.999), 'eps': 1e-7},
                'scheduler_name'  : 'StepLR',
                'scheduler_kwargs': {'eta_min': 2e-4, 'T_max': 300, 'gamma': 0.82, 'step_size': 1},
                'epochs'          : 300,
                'batch_size'      : 256,
                'iteration_test'  : (118272 // 256) // 16,
                'iteration_train' : 118272 // 256,
            }
        torch.cuda.set_device(2)
        if self.wandb_init:
            wandb.login(key='84d6a92704bf4bf19d2ecc87a55eea5ce77a8725')
            self.run = wandb.init(project='Homework_3', config=self.config)
            self.model_name = 'run_' + self.run.name + '_model'
        else:
            self.model_name = 'testing'
        self.start_epoch = 0
        self.num_epochs  = self.config['epochs']
        
    def __init__(self, wandb_init=True):
        # Get  files
        self.get_files()
        self.set_wandb(wandb_init)
        self.set_net()
        self.set_opt_sched()
        # Initialize all essentials critetia and metrics
        self.SoftMax  = torch.nn.Softmax(dim=1)
        self.entropy = torch.nn.CrossEntropyLoss() 
        self.best_loss = 1e8
        
        
    def wandb_log(self, config, name):    
        wandb.log({f'{name} loss'        : np.mean(config['loss']),
                   f'{name} LR'          : float(get_lr(self.optimizer)),
                   'Epoch'               : config['i_epoch']})
    
    def compute_loss(self, image_features, captions_ix):
        """
        :param image_features: torch tensor containing VGG features. shape: [batch, cnn_channels, width * height]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
            padded with pad_ix
        :returns: crossentropy (neg llh) loss for next captions_ix given previous ones plus
              attention regularizer. Scalar float tensor
        """
        pad_ix = 3
        image_features, captions_ix = image_features.cuda(), captions_ix.cuda()
        # captions for input - all except last cuz we don't know next token for last one.
        captions_ix_inp = captions_ix[:, :-1].contiguous()
        captions_ix_next = captions_ix[:, 1:].contiguous()

        # apply the network, get predictions, attnetion map and gates for captions_ix_next
        logits_for_next, attention_map = self.net.forward(image_features, captions_ix_inp)


        # compute the loss function between logits_for_next and captions_ix_next
        # Use the mask, Luke: make sure that predicting next tokens after EOS do not contribute to loss
        # you can do that either by multiplying elementwise loss by (captions_ix_next != pad_ix)
        # or by using ignore_index in some losses.

        logits_for_next = logits_for_next.transpose(1, 2)    
        #captions_ix_next = torch.nn.functional.one_hot(captions_ix_next, num_classes=logits_for_next.shape[1])
        #captions_ix_next = captions_ix_next.transpose(1, 2).float()
        
        #loss = self.entropy(logits_for_next, captions_ix_next)
        loss = torch.nn.functional.cross_entropy(logits_for_next,captions_ix_next, ignore_index=self.pad_ix)
        # the regularizer for attention - this one requires the attention over each position to sum up to 1,
        # i. e. to look at the whole image during sentence generation process
        
        mask = (captions_ix_inp != pad_ix)
        masked_attention_map = attention_map * mask[:, :, None].float()
        regularizer = ((1 - masked_attention_map.sum(1)) ** 2).mean()

        return loss + regularizer
    
    def test(self, i_epoch):
        # Test
        self.net.eval()
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        
        with torch.no_grad():
            loop = tqdm(range(self.config['iteration_test']), total=self.config['iteration_test'], leave=False)
            for batch_idx in loop:
                loss = self.compute_loss(*self.generate_batch(self.test_img_codes, self.test_captions, self.config['batch_size']))
                config['loss'].append(loss.item())
                # LOOPA and PUPA
                loop.set_description(f"Epoch (Test)[{i_epoch}/{self.num_epochs}]")
                loop.set_postfix(loss=np.nanmean(config['loss']))
                gc.collect()
                torch.cuda.empty_cache()
                
            config['i_epoch'] = i_epoch
            if self.wandb_init:
                self.wandb_log(config, name='Test')

        # Save checkpoint.
        loss = np.mean(config['loss'])
        if loss < self.best_loss:
            self.best_loss = loss
            self.save_model(i_epoch, self.model_name)
    
    def train(self, i_epoch):
        # Train
        config = self.get_config()
        self.net.train()
        loop = tqdm(range(self.config['iteration_train']), total=self.config['iteration_train'], leave=False)
        for batch_idx in loop:
            loss = self.compute_loss(*self.generate_batch(self.train_img_codes, self.train_captions, self.config['batch_size']))
            config['loss'].append(loss.item())
            # Make backward step
            loss.backward()
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            # LOOPA and PUPA
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(loss=np.mean(config['loss']))
            
            gc.collect()
            torch.cuda.empty_cache()
        config['i_epoch'] = i_epoch
        if self.wandb_init:
            self.wandb_log(config, name='Train')
            
    def fit(self):        
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.test(i_epoch)        
            self.scheduler.step()
        if self.wandb_init:
            self.run.finish()
        
    def set_opt_sched(self):
        #self.optimizer = optim.AdamW(
        #    params       = self.net.parameters(),
        #    lr           = self.config['learning_rate'],
        #    betas        = self.config['optimizer_kwargs']['betas'],
        #    eps          = self.config['optimizer_kwargs']['eps'],
        #    weight_decay = self.config['weight_decay']
        #   )
        self.optimizer = optim.SGD(
            params       = self.net.parameters(),
            lr           = self.config['learning_rate'],
            weight_decay = self.config['weight_decay'],
            momentum     = self.config['momentum']
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['scheduler_kwargs']['step_size'],
            gamma=self.config['scheduler_kwargs']['gamma']
        )
            
    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.net = CaptionNet(
            n_tokens     = self.config['n_tokens'], 
            emb_size     = self.config['emb_size'], 
            lstm_units   = self.config['lstm_units'], 
            cnn_channels = self.config['cnn_channels'])
        
        self.net = self.net.to(self.device)
        if self.wandb_init:
            wandb.watch(self.net, log_freq=100)
     
    def load(self, filename):
        
        checkpoint = torch.load(f'/media/longlake_student/lungs/models/university/{filename}.pth')
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net          = self.net.to(self.device)
        self.best_loss    = checkpoint['loss']
        self.start_epoch  = checkpoint['epoch']    
        
        if self.wandb_init:
            wandb.watch(self.net, log_freq=100)
        
    def save_model(self, i_epoch, name):
        #print('Saving...')
        state = {
            'net'      : self.net.state_dict(), 
            'loss'     : self.best_loss, 
            'epoch'    : i_epoch, 
            }
        torch.save(state, f'/media/longlake_student/lungs/models/university/{name}.pth')
        print(f'Saving... Epoch[{i_epoch}]')
    
    def generate_batch(self, img_codes, captions, batch_size, max_caption_len=None):
        #sample sequential numbers for image/caption indicies (for trainign speed up)
        
        random_image_ix = np.arange(batch_size, dtype='int') + self.last_batch_end.get(len(img_codes), 0)
        self.last_batch_end[len(img_codes)] = self.last_batch_end.get(len(img_codes), 0) + batch_size
        if self.last_batch_end[len(img_codes)] + batch_size >= len(img_codes):
            self.last_batch_end[len(img_codes)] = 0

        #get images
        batch_images = np.vstack([img_codes[i][None] for i in random_image_ix])
        batch_images = batch_images.reshape(batch_images.shape[0], batch_images.shape[1], -1)

        #5-7 captions for each image
        captions_for_batch_images = captions[random_image_ix]

        #pick one from a set of captions for each image
        batch_captions = list(map(choice, captions_for_batch_images))
        

        #convert to matrix
        batch_captions_ix = self.as_matrix(batch_captions, max_len=max_caption_len)

        return torch.tensor(batch_images, dtype=torch.float32), torch.tensor(batch_captions_ix, dtype=torch.int64)
    
    def get_files(self):
        self.last_batch_end = {}
        
        f = h5py.File('./attention/img_codes.hdf5', 'r')
        img_codes = f['data']

        captions = json.load(open('./attention/captions_tokenized.json'))
        for img_i in range(len(captions)):
            for caption_i in range(len(captions[img_i])):
                sentence = captions[img_i][caption_i]
                captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]
        
        
        word_counts = Counter()
        for cap in captions: 
            for sent in cap:
                for word in sent:
                    word_counts[word] += 1
        vocab  = ['#UNK#', '#START#', '#END#', '#PAD#']
        vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
        self.n_tokens = len(vocab)

        #assert 10000 <= n_tokens <= 10500
        self.word_to_index = {w: i for i, w in enumerate(vocab)}
        self.eos_ix = self.word_to_index['#END#']
        self.unk_ix = self.word_to_index['#UNK#']
        self.pad_ix = self.word_to_index['#PAD#']

        
        captions = np.array(captions, dtype=object)
        np.random.seed(42)
        perm = np.random.permutation(len(img_codes))
        threshold = round(len(img_codes) * 0.1)
        
        train_img_idx, test_img_idx = perm[threshold:], perm[: threshold]
        train_img_idx.sort()
        test_img_idx.sort()
        
        self.train_img_codes = IdxDataset(img_codes, train_img_idx)
        self.test_img_codes  = IdxDataset(img_codes, test_img_idx)
        self.train_captions  = IdxDataset(captions, train_img_idx)
        self.test_captions   = IdxDataset(captions, test_img_idx)
    
    def as_matrix(self, sequences, max_len=None):
        """ Convert a list of tokens into a matrix with padding """
        max_len = max_len or max(map(len,sequences))

        matrix = np.zeros((len(sequences), max_len), dtype='int32') + self.pad_ix
        for i,seq in enumerate(sequences):
            row_ix = [self.word_to_index.get(word, self.unk_ix) for word in seq[:max_len]]
            matrix[i, :len(row_ix)] = row_ix

        return matrix

    
    def get_config(self):
        return {
            'loss'   : [],
            'i_epoch': 0,
            }