import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class compressor(nn.Module):
    def __init__(self, args):
        super(compressor, self).__init__()
        self.compressor = simple_align(args)

    def forward(self, x):
        return self.compressor(x)

class simple_align(nn.Module):
    def __init__(self, args) -> None:
        super(simple_align, self).__init__()
        c1 = args["in_channel"]
        c2 = args["channel"]
        h1 = args["in_size"][0]
        w1 = args["in_size"][1]
        h2 = args["featrue_size"][0]
        w2 = args["featrue_size"][1]
        if h1 // h2 >=2 :
            self.max_pooling = nn.MaxPool2d(kernel_size=h1 // h2)
        else:
            self.max_pooling =  nn.Upsample(size=(h2, w2), mode='nearest')
        self.conv_downsampling=nn.Conv2d(in_channels=c1,out_channels=c2,kernel_size=1)

    def forward(self, x):
        x = self.conv_downsampling(x)
        x = self.max_pooling(x)
        return x

 
class simple_align_v2(nn.Module):
    def __init__(self, args) -> None:
        super(simple_align_v2, self).__init__()
        c1 = args["in_channel"]
        c2 = args["channel"]
        h1 = args["in_size"][0]
        w1=args["in_size"][1]
        h2 = args["featrue_size"][0]
        w2=args["featrue_size"][1]
        
        self.size_reshaper = nn.Upsample(size=(h2, w2), mode='nearest')

        self.conv_downsampling=nn.Conv2d(in_channels=c1,out_channels=c2,kernel_size=1)


    def forward(self, x):
        x = self.conv_downsampling(x)
        
        x = self.size_reshaper(x)

        return x


class coarse_align(nn.Module):
    def __init__(self, args) -> None:
        super(coarse_align, self).__init__()
        c1 = args["c_k"]
        c2 = args["channel"]
        c3 = args["c_q"]
        h1 = args["f_k"][0]
        h2 = args["featrue_size"][0]
        h3 = args["f_q"][0]
        kernel_size = args["kernel_size"]
        padding = args["padding"]
        assert h1 >= h2 and h3 >= h2
        self.conv_downsampling1 = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=kernel_size,
            stride=h1 // h2,
            padding=padding,
        )
        self.conv_downsampling2 = nn.Conv2d(
            in_channels=c3,
            out_channels=c2,
            kernel_size=kernel_size,
            stride=h3 // h2,
            padding=padding,
        )

    def forward(self, x, y):
        y = self.conv_downsampling1(y)
        x = self.conv_downsampling2(x)
        return x, y