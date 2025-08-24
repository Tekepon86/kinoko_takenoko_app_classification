import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
import torch
from torchvision.models import resnet18

class CNN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        n_features = self.feature.fc.in_features
        self.feature.fc = nn.Identity()  # 出力をそのまま取り出す、最後の全結合層を取り外す

        #新しい全結合層を追加
        self.fc = nn.Linear(n_features, 2)
        #for param in self.feature.parameters():
        #    param.requires_grad = False

        #self.fc = nn.Linear(512, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
