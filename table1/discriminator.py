import torch
import torch.nn as nn

class StCDiscriminator(nn.Module):

    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        super(StCDiscriminator, self).__init__()
        layers = [nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
