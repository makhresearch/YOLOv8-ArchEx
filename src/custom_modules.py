import torch
import torch.nn as nn
# To access the autopad function, we need to import it from the modules folder
from .modules.conv import autopad

class MyCustomConv(nn.Module):
    """
    This is an example of a custom convolution module.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """A helper method to improve inference speed."""
        return self.act(self.conv(x))