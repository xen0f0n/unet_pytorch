import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetOrig(nn.Module):
    '''
    Implements Original UNet --> https://arxiv.org/pdf/1505.04597v1.pdf
    '''

    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.description = "Implements Original UNet --> https://arxiv.org/pdf/1505.04597v1.pdf"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = DoubleConv_ReLU(n_channels, 64)
        self.conv2 = DoubleConv_ReLU(64, 128)
        self.conv3 = DoubleConv_ReLU(128, 256)
        self.conv4 = DoubleConv_ReLU(256, 512)
        self.conv5 = DoubleConv_ReLU(512, 1024)
        self.up6 = Up(1024, 512)
        self.up7 = Up(512, 256)
        self.up8 = Up(256, 128)
        self.up9 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)
        conv4 = self.conv4(pool3)
        drop4 = F.dropout2d(conv4, 0.5)
        pool4 = F.max_pool2d(drop4, kernel_size=2, stride=2)

        conv5 = self.conv5(pool4)
        drop5 = F.dropout2d(conv5, 0.5)

        up6 = self.up6(drop5, drop4)
        up7 = self.up7(up6, conv3)
        up8 = self.up8(up7, conv2)
        up9 = self.up9(up8, conv1)
        logits = self.outc(up9)
        return logits

class DoubleConv_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_ReLU(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv_ReLU(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)