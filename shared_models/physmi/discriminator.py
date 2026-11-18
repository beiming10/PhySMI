# discriminator.py

import torch
import torch.nn as nn

class StCDiscriminator(nn.Module):
    """
    PatchGAN判别器，适用于判断单通道的StC图像是否真实。
    它的任务是区分 "模型生成的abundance通道" 和 "StC库里的真实结构图像"。

    输入: 一个单通道的图像 (B, 1, H, W)。
    输出: 一个特征图 (B, 1, H', W')，其中每个值代表原图一个patch的真实性。
    """
    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        """
        Args:
            in_channels (int): 输入图像的通道数。对于StC，这里固定为1。
            ndf (int): 第一个卷积层的特征数。
            n_layers (int): 判别器中的降采样卷积层数量。
        """
        super(StCDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # 最终输出层
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入的单通道StC图像，形状为 [B, 1, H, W]。
        """
        return self.model(x)