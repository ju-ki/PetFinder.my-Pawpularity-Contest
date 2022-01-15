#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'nvidia-smi')


# In[2]:


import os
import sys

def set_environment() -> None:
    IN_COLAB = 'google.colab' in sys.modules
    IN_KAGGLE = 'kaggle_web_client' in sys.modules
    LOCAL = not (IN_KAGGLE or IN_COLAB)
    print(f'IN_COLAB:{IN_COLAB}, IN_KAGGLE:{IN_KAGGLE}, LOCAL:{LOCAL}')

    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        get_ipython().system(u'wget -O fmix.zip https://github.com/ecs-vlc/fmix/archive/master.zip')
        get_ipython().system(u'unzip -qq fmix.zip')
        get_ipython().system(u'mv FMix-master/* ./')
        get_ipython().system(u'rm -r FMix-master')
        if not os.path.exists("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/data/input"):
            os.chdir("/content/drive/MyDrive/jukiya/")
            get_ipython().system(u'pip install --quiet kaggle')
            get_ipython().system(u'mkdir -p ~/.kaggle')
            get_ipython().system(u'cp kaggle.json ~/.kaggle/')
            get_ipython().system(u'chmod 600 /root/.kaggle/kaggle.json')
            os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/data/")
            get_ipython().system(u'mkdir input')
            os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest//data/input/")
            # !kaggle competitions download -c petfinder-pawpularity-score
        os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest//")
        get_ipython().system(u'pip install --quiet -r requirements.txt')
        get_ipython().system(u'pip uninstall --quiet -y albumentations')
        get_ipython().system(u'pip install --quiet albumentations==1.1.0')
        get_ipython().system(u'pip install --quiet -q pytorch-lightning')
        get_ipython().system(u'pip install --quiet torch_optimizer==0.1.0')
        get_ipython().system(u'pip install --quiet timm')
        get_ipython().system(u'pip install --quiet grad-cam')
        get_ipython().system(u'pip install -q ttach')
        get_ipython().system(u'pip install -q lightly')
    if IN_KAGGLE:
        pass
    return IN_COLAB, IN_KAGGLE, LOCAL


# In[3]:


IN_COLAB, IN_KAGGLE, LOCAL = set_environment()


# In[4]:


import cv2
from PIL import Image

import os
import gc
import sys
import json
import math
import random
import time
# from time import time
from datetime import datetime
from collections import Counter, defaultdict

import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set(style="whitegrid")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from tqdm.auto import tqdm
# import category_encoders as ce

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

if IN_KAGGLE:
    sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
    sys.path.append("../input/image-fmix/FMix-master")
    from fmix import sample_mask
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

if IN_COLAB:
    os.chdir("/content/")
    from fmix import sample_lam, sample_mask
    os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/")
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


class Config:
    apex = False
    DEBUG = False
    print_freq = 150
    num_workers = 4
    size = 384
    batch_size = 32
    exp_name = "exp100"
    model_name = "tf_efficientnetv2_b0"
    # model_name = "resnet50d"
    optimizer_name = "AdamW"
    scheduler = "CosineAnnealingWarmRestarts"
    epochs = 20
    #num_warmup_steps=100 # ['linear', 'cosine']
    #num_cycles=0.5 # 'cosine'
    factor=0.8 # ReduceLROnPlateau
    patience=2 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max = 10 #CosineAnnealingLR
    T_0 = 20 #CosineAnnealingWarmRestarts
    # T_mult = 1 #CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 1
    target_col = "Pawpularity"
    n_folds = 10
    trn_fold = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mixup = True
    cutmix = False
    fmix = False
    train = True
    grad_cam = False
    inference = False
if Config.DEBUG:
    Config.n_folds = 2
    Config.trn_fold = [0, 1]
    Config.epochs = 2

if IN_COLAB:
    import requests
    def get_exp_name():
        return requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"].split("_")[0]
    Config.exp_name = get_exp_name()
    print(Config.exp_name)


# In[6]:


if IN_COLAB:
    os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/code/")
    from util.AverageMeter import AverageMeter, asMinutes, timeSince
    from util.logger import Logger
    from pytorch_model.util import get_optimizer, get_scheduler
    os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/")
    LOG = "./log/"
    SUBMISSION = "./data/submission/"
    # if not os.path.join(f"/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/models/{Config.exp_name}"):
    print("Creating folder")
    os.chdir("./models/")
    get_ipython().system(u'mkdir {Config.exp_name}')
    os.chdir("/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/")
    OUTPUT = f"/content/drive/MyDrive/PetFinder-my-Pawpularity-Contest/models/{Config.exp_name}"
    # OUTPUT = f"./models/{Config.exp_name}"
elif IN_KAGGLE:
    import math
    import time


    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(since, percent):
        now = time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
    
    import os
    import datetime
    import logging


    class Logger:
        """save log"""

        def __init__(self, path, exp_name=None):
            self.general_logger = logging.getLogger(path)
            stream_handler = logging.StreamHandler()
            file_general_handler = logging.FileHandler(
                os.path.join(path, f'{exp_name}.log'))
            if len(self.general_logger.handlers) == 0:
                self.general_logger.addHandler(stream_handler)
                self.general_logger.addHandler(file_general_handler)
                self.general_logger.setLevel(logging.INFO)

        def info(self, message):
            # display time
            self.general_logger.info(
                '[{}] - {}'.format(self.now_string(), message))

        @staticmethod
        def now_string():
            return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    LOG = "./"
    OUTPUT = "./"
    SUBMISSION = "./"
    
if Config.train:    
    logger = Logger(LOG, exp_name=Config.exp_name)


# In[7]:


if IN_COLAB:
    train_df = pd.read_csv("./data/external/clean_petfinder_dataset/train.csv")
    test_df = pd.read_csv("./data/input/test.csv")
elif IN_KAGGLE:
    train_df = pd.read_csv("../input/petfinder-pawpularity-score-clean/train.csv")
    test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
if Config.DEBUG:
    train_df = train_df.sample(n=1000, random_state=Config.seed).reset_index(drop=True) 

print(train_df.shape)


# In[8]:


def get_train_file_path(image_id):
    if IN_COLAB:
        return "./data/external/clean_petfinder_dataset/train/{}.jpg".format(image_id)
    elif IN_KAGGLE:
        return "../input/petfinder-pawpularity-score-clean/{}.jpg".format(image_id)

def get_test_file_path(image_id, is_colab=False):
    if IN_COLAB:
        return "./data/external/clean_petfinder_dataset/test/{}.jpg".format(image_id)
    elif IN_KAGGLE:
        return "../input/petfinder-pawpularity-score/test/{}.jpg".format(image_id)


train_df["file_path"] = train_df["Id"].apply(lambda x: get_train_file_path(x))
test_df["file_path"] = test_df["Id"].apply(lambda x: get_test_file_path(x))
print(train_df.shape)


# In[9]:


def remove_worse_abs_error_data(df):
    df = df[(df["Id"] != "f4dd9ea389798d703ee5ebe2a82c8653") & (df["Id"] != "4c1e05895368c997fdd709bbd1ac3dae") & (df["Id"] != "e638853bdec13e6cc76b69f169b34740") &
            (df["Id"] != "3d69187b44aba9adbd7fd5f0d3554cd1") & (df["Id"] != "941e532a21f7edb21766e88a15e588cb") & (df["Id"] != "d00e5d29d4ec09de1e0e0ebec41b2d03") &
            (df["Id"] != "2de3bf4fdcefa2f3c5e5b77fa1e3d262") & (df["Id"] != "346650fd7c4fcdcbd7a119436881e52a") & (df["Id"] != "b2a389311b683a90c3c9763540a86bab") & 
            (df["Id"] != "054cef9194f1a4513dc0965893b589bf") & (df["Id"] != "e5a84da44afc0abddfbb36907a5ee845") & (df["Id"] != "deb8d1618d0cdc2a78c8198b882ebb2b")]
    return df.reset_index(drop=True)

train_df = remove_worse_abs_error_data(train_df)
print(train_df.shape)


# In[10]:


def get_score(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
    return score

def seed_everything(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
seed_everything(Config.seed)


# In[11]:


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df[Config.target_col].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        # label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx]).float()
        # image = torch.tensor(image, dtype=torch.float32)
        return image, label

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    
class GradCAMDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.image_ids = df['Id'].values
        self.file_names = df['file_path'].values
        self.labels = df[Config.target_col].values
        self.transform = get_transforms(phase='valid')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image = cv2.resize(image, (Config.size, Config.size)).copy()
        if self.transform:
            image = self.transform(image=image)['image']
        # label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx]).float()
        # image = torch.tensor(image, dtype=torch.float32)
        return image_id, image, vis_image, label


# In[12]:


def get_transforms(*, phase):
    if phase == "train":
        return A.Compose([
                          A.RandomResizedCrop(Config.size, Config.size, scale=(0.85, 1.0)),
                          A.RandomBrightnessContrast(),
                          A.HorizontalFlip(),
                        #   A.VerticalFlip(),
                          A.ShiftScaleRotate(),
                          A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                        #   A.CoarseDropout(p=0.5),
                        #   A.Cutout(p=0.5),
                        #   A.FancyPCA(),
                        #   A.ColorJitter(),
                        # A.CoarseDropout(),
                        #   A.ToGray(),
                          A.GaussianBlur(p=0.5, sigma_limit=(0.1, 2.0)),
                          A.Normalize(
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225],
                          ),
                          ToTensorV2(),
        ])
    elif phase == "valid":
        return A.Compose([
                          A.Resize(Config.size, Config.size),
                          A.Normalize(
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]
                          ),
                          ToTensorV2(),
        ])


# In[13]:


if Config.train:
    num_bins = int(np.floor(1 + np.log2(len(train_df))))
    train_df["bins"] = pd.cut(train_df[Config.target_col], bins=num_bins, labels=False)
    Fold = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df["bins"])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype("int16")


# In[14]:


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained)
        # self.model = make_simsiam_model()
        if cfg.model_name == "tf_efficientnet_b0_ns":
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            # self.fc = nn.Linear(self.n_features, self.cfg.target_size)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.n_features, self.cfg.target_size))
        elif cfg.model_name == "resnet50d":
            self.n_features = self.model.get_classifier().in_features
            self.model.reset_classifier(0)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.n_features, self.cfg.target_size))
        elif cfg.model_name == "tf_efficientnetv2_b0":
            self.n_features = self.model.get_classifier().in_features
            self.model.reset_classifier(0)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.n_features, self.cfg.target_size))
        else:
            self.n_features = self.model.head.in_features
            self.model.reset_classifier(0)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.n_features, self.cfg.target_size))

        # self._init_weight()

    def _init_weight(self):
        for name, p in self.named_parameters():
            if "fc" in name:
                if 'weight' in name:
                    p = p.reshape(1, -1)#fat in fat out errorの対策。何故かlstmのweight_ihのshapeが整っているのが謎
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def feature(self, image):
        feature = self.model(image)
        return feature
        
    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output


# In[15]:


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def mixup(x: torch.Tensor, y:torch.Tensor, alpha:float = 1.0):
    assert alpha > 0
    assert x.size(0) > 1

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

def mixup_criterion(criterion, pred, target_a, target_b, lam):
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)

def cutmix_criterion(criterion, pred, target_a, target_b, lam):
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# def cutmix(data, target, alpha=1.0):
#     # indices = torch.randperm(data.size(0))
#     # shuffled_data = data[indices]
#     # shuffled_target = target[indices]

#     lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
#     new_data = data.clone()
#     new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
#     # targets = (target, shuffled_target, lam)

#     return new_data, target, shuffled_data, lam

def cutmix(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, bbx1:bbx2, bby1:bby2] = x[index, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    target_a, target_b = y, y[index]
    return x, target_a, target_b, lam

def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    #mask =torch.tensor(mask, device=device).float()
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    # targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets, shuffled_targets, lam


# In[16]:


class EarlyStopping:
    
    def __init__(self, patience=2, seq=False, logger=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.logger = logger
        
    def __call__(self, loss, model, preds, path):
        if self.best_score is None:
            self.best_score = loss
            self.save_checkpoint(model, preds, path)
        elif loss < self.best_score:
            self.logger.info(f'Loss decreased {self.best_score:.5f} -> {loss:.5f}')
            self.best_score = loss
            self.counter = 0
            self.save_checkpoint(model, preds, path)
        else:
            self.counter += 1
            if self.counter > self.patience: self.stop = True
                
    def save_checkpoint(self, model,  preds, path):
        save_list = {'model': model.state_dict(), 
                     'preds': preds}
        SAVE_PATH = path
        torch.save(save_list, SAVE_PATH)


# In[17]:


if IN_KAGGLE:
    def get_scheduler(optimizer, Config, num_train_steps=None):
        """
        if use get_linear_schedule_with_warmup or get_cosine_schedule_with_warmup, set num_train_steps.
        ex:
        num_train_steps = int(len(train_folds) / Config.batch_size * Config.epochs)
        if cosine or linear:
            num_warmup_steps (int) – The number of steps for the warmup phase.
            num_training_steps (int) – The total number of training steps.
            cosine only:
            num_cycles (float, optional, defaults to 0.5) – The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).
        if ReduceLROnPlateau:
            mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing
            factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            patience (int) – Number of epochs with no improvement after which learning rate will be reduced.
        if CosineAnnealingLR:
            T_max (int) – Maximum number of iterations
            eta_min (float) – Minimum learning rate. Default: 0.
        if CosineAnnealingWarmRestarts:
            T_0 (int) – Number of iterations for the first restart
            T_mult (int, optional) – A factor increases T_{i}T after a restart. Default: 1.
            eta_min (float, optional) – Minimum learning rate. Default: 0.
        """
        if Config.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif Config.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=Config.num_cycles
            )
        elif Config.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=Config.factor, patience=Config.patience, verbose=True, eps=Config.eps)
        elif Config.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
        elif Config.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=Config.T_0, T_mult=1, eta_min=Config.min_lr, last_epoch=-1)
        else:
            raise NotImplementedError
        return scheduler


    def get_optimizer(model: nn.Module, Config: dict):
        if Config.optimizer_name == "AdamW":
            optimizer = AdamW(model.parameters(), lr=Config.lr,
                            weight_decay=Config.weight_decay)
        elif Config.optimizer_name == "Adam":
            optimizer = Adam(model.parameters(), lr=Config.lr,
                            weight_decay=Config.weight_decay)
        elif Config.optimizer_name == "SGD":
            optimizer = SGD(model.parameters(), lr=Config.lr,
                            weight_decay=Config.weight_decay)
        else:
            raise Exception('Unknown optimizer: {}'.format(Config.optimizer_name))
        return optimizer


# In[18]:


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    if Config.apex:
        scaler = GradScaler()
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        mixup_decision = np.random.rand()
        if mixup_decision < 0.5:
            if Config.mixup:
                images, target_a, target_b, lam = mixup(images, labels, alpha=1.0)
                target_a, target_b = target_a.to(device), target_b.to(device)
        # elif mixup_decision >= 0.25 and mixup_decision < 0.5:
        #     if Config.fmix:
        #         images, target_a, target_b, lam = fmix(images, labels, alpha=0.4, decay_power=5.0, shape=(Config.size, Config.size))
        #         target_a, target_b = target_a.to(device), target_b.to(device)
        #     elif Config.cutmix:
        #         images, target_a, target_b, lam = cutmix(images, labels)
        #         target_a, target_b = target_a.to(device), target_b.to(device)

        if Config.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds.view(-1), labels)
        else:
            y_preds = model(images.float())
            if mixup_decision < 0.5:
                if Config.mixup:
                    loss = mixup_criterion(criterion, y_preds.view(-1), target_a, target_b, lam)
                elif Config.cutmix:
                    loss = cutmix_criterion(criterion, y_preds.view(-1), target_a, target_b, lam)
                elif Config.fmix:
                    loss = mixup_criterion(criterion, y_preds.view(-1), target_a, target_b, lam)
                else:
                    loss = criterion(y_preds.view(-1), labels)
            else:
                loss = criterion(y_preds.view(-1), labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        if Config.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        del loss
        gc.collect()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            if Config.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        end = time.time()
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    all_labels = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(torch.sigmoid(y_preds).to('cpu').numpy() * 100)
        all_labels.append(labels.detach().cpu().numpy()*100)
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        del loss 
        gc.collect()
        end = time.time()
    predictions = np.concatenate(preds)
    all_labels = np.concatenate(all_labels)
    return losses.avg, predictions, all_labels


# In[19]:


def get_grad_cam(model, device, x_tensor, img, label, plot=False):
    result = {"vis": None, "img": None, "pred": None, "label": None}
    with torch.no_grad():
        pred = model(x_tensor.unsqueeze(0).to(device))
    pred = np.concatenate(pred.to('cpu').numpy())[0]
    target_layer = model.modules.features
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available())
    output = cam(input_tensor=x_tensor.unsqueeze(0).to(device))
    try:
        vis = show_cam_on_image(img / 255., output[0])
    except:
        return result
    if plot:
        fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
        axes[0].imshow(vis)
        axes[0].set_title(f"pred={pred:.4f}")
        axes[1].imshow(img)
        axes[1].set_title(f"target={label}")
        plt.show()
    result = {"vis": vis, "img": img, "pred": pred, "label": label}
    torch.cuda.empty_cache()
    return result


# In[20]:


def train_loop(train_df, fold):
    
    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    folds = train_df.copy()

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    folds["Pawpularity"] = folds["Pawpularity"] / 100

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    # valid_labels = valid_folds[Config.target_col].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(phase='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(phase='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.batch_size, 
                              shuffle=True, 
                              num_workers=Config.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=Config.batch_size * 2, 
                              shuffle=False, 
                              num_workers=Config.num_workers, pin_memory=True, drop_last=False)
    
    early_stopping = EarlyStopping(patience=3, logger=logger)
    
    # ====================================================
    # scheduler 
    # ====================================================
    # def get_scheduler(optimizer):
    #     if Config.scheduler=='ReduceLROnPlateau':
    #         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.factor, patience=Config.patience, verbose=True, eps=Config.eps)
    #     elif Config.scheduler=='CosineAnnealingLR':
    #         scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
    #     elif Config.scheduler=='CosineAnnealingWarmRestarts':
    #         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=1, eta_min=Config.min_lr, last_epoch=-1)
    #     return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(Config, pretrained=True)
    model.to(device)
    
    optimizer = get_optimizer(model, Config)
    scheduler = get_scheduler(optimizer, Config)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(Config.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds, valid_labels = valid_fn(valid_loader, model, criterion, device)
        

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s, lr: {optimizer.param_groups[0]["lr"]:.5f}')
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        if score < best_score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT+f'/{Config.model_name}_fold{fold}_best.pth')

        early_stopping(score, model, preds, path=OUTPUT + f"/{Config.model_name}_fold{fold}_best.pth")

        if early_stopping.stop:
            logger.info("Early stop!!")
            break

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
    
    valid_folds['preds'] = torch.load(OUTPUT+f'/{Config.model_name}_fold{fold}_best.pth', 
                                      map_location=torch.device('cpu'))['preds']
    valid_folds["Pawpularity"] = valid_folds["Pawpularity"] * 100

    return valid_folds


# In[21]:


def main():

    """
    Prepare: 1.train 
    """

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[Config.target_col].values
        score = get_score(labels, preds)
        logger.info(f'Score: {score:<.4f}')
    
    if Config.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(Config.n_folds):
            if fold in Config.trn_fold:
                _oof_df = train_loop(train_df, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        logger.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT+'/oof_df.csv', index=False)
    
    if Config.grad_cam:
        for fold in range(Config.n_folds):
            if fold in Config.trn_fold:
                # load model
                model = CustomModel(Config, pretrained=False)
                state = torch.load(OUTPUT+f'/{Config.model_name}_fold{fold}_best.pth', 
                                   map_location=torch.device('cpu'))['model']
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                # load oof
                oof = pd.read_csv(OUTPUT+'/oof_df.csv')
                oof['diff'] = abs(oof['Pawpularity'] - oof['preds'])
                oof = oof[oof['fold'] == fold].reset_index(drop=True)
                # grad-cam (oof ascending=False)
                count = 0
                oof = oof.sort_values('diff', ascending=False)
                valid_dataset = GradCAMDataset(oof)
                for i in range(len(valid_dataset)):
                    image_id, x_tensor, img, label = valid_dataset[i]
                    result = get_grad_cam(model, device, x_tensor, img, label, plot=True)
                    if result["vis"] is not None:
                        count += 1
                    if count >= 5:
                        break
                # grad-cam (oof ascending=True)
                count = 0
                oof = oof.sort_values('diff', ascending=True)
                valid_dataset = GradCAMDataset(oof)
                for i in range(len(valid_dataset)):
                    image_id, x_tensor, img, label = valid_dataset[i]
                    result = get_grad_cam(model, device, x_tensor, img, label, plot=True)
                    if result["vis"] is not None:
                        count += 1
                    if count >= 5:
                        break
    if Config.inference:
        test_dataset = TestDataset(test_df, transform=get_transforms(phase='valid'))
        test_loader = DataLoader(test_dataset, 
                         batch_size=Config.batch_size * 2, 
                         shuffle=False, 
                         num_workers=Config.num_workers, pin_memory=True, drop_last=False)
        def inference_fn(test_loader, model, device):
            model.eval()
            preds = []
            tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
            for step, (images) in tk0:
                images = images.to(device)
                batch_size = images.size(0)
                with torch.no_grad():
                    pred = model(images)
                preds.append(torch.sigmoid(pred).view(-1).cpu().detach().numpy() * 100)
            preds = np.concatenate(preds)
            return preds

        final_pred = []

        for fold in range(Config.n_folds):
            model = CustomModel(Config, pretrained=False)
            if IN_COLAB:
                state = torch.load(OUTPUT + f"/{Config.exp_name}/{Config.model_name}_fold{fold}_best.pth", 
                                   map_location=torch.device('cpu'))['model']                
            elif IN_KAGGLE:
                state = torch.load(f"../input/my-pf-dataset/{Config.exp_name}/{Config.model_name}_fold{fold}_best.pth", 
                                   map_location=torch.device('cpu'))['model']
            model.load_state_dict(state)
            model.to(device)
            preds = inference_fn(test_loader, model, device)
            final_pred.append(preds)
            del state; gc.collect()
            torch.cuda.empty_cache()
        
        final_pred = np.mean(np.column_stack(final_pred), axis=1)
        print(final_pred)
        if IN_KAGGLE:
            sub_df = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
            sub_df["Pawpularity"] = final_pred
            sub_df.to_csv("submission.csv", index=False)
            display(sub_df.head())
    


# In[22]:


if __name__ == '__main__':
    main()


# In[22]:




