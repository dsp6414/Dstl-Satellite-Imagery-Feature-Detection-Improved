import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip1 = self.pool(x)
        skip1 = F.relu(self.conv3(skip1))
        skip1 = F.relu(self.conv4(skip1))
        skip1 = F.relu(self.conv5(skip1))
        skip1 = self.upsample(skip1)
        x = torch.cat([x, skip1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return F.sigmoid(x)


class MediumUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(128+64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv9 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.convfinal = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        skip0 = F.relu(self.conv2(x))
        skip1 = self.pool1(x)
        skip1 = F.relu(self.conv3(skip1))
        skip1 = F.relu(self.conv4(skip1))
        skip2 = self.pool2(skip1)
        skip2 = F.relu(self.conv5(skip2))
        skip2 = F.relu(self.conv6(skip2))
        x = self.upsample1(skip2)
        x = torch.cat([x, skip1], 1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.upsample2(x)
        x = torch.cat([x, skip0], 1)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.convfinal(x)
        return F.sigmoid(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv11 = nn.Conv2d(512+256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv13 = nn.Conv2d(256+128, 128, 3, padding=1)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv15 = nn.Conv2d(128+64, 64, 3, padding=1)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv17 = nn.Conv2d(64+32, 32, 3, padding=1)
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1)
        self.convfinal = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        skip0 = F.relu(self.conv2(x))
        skip1 = self.pool(skip0)
        skip1 = F.relu(self.conv3(skip1))
        skip1 = F.relu(self.conv4(skip1))
        skip2 = self.pool(skip1)
        skip2 = F.relu(self.conv5(skip2))
        skip2 = F.relu(self.conv6(skip2))
        skip3 = self.pool(skip2)
        skip3 = F.relu(self.conv7(skip3))
        skip3 = F.relu(self.conv8(skip3))
        skip4 = self.pool(skip3)
        skip4 = F.relu(self.conv9(skip4))
        skip4 = F.relu(self.conv10(skip4))
        x = self.upsample(skip4)
        x = torch.cat([x, skip3], 1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.upsample(x)
        x = torch.cat([x, skip2], 1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = self.upsample(x)
        x = torch.cat([x, skip1], 1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.upsample(x)
        x = torch.cat([x, skip0], 1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = self.convfinal(x)
        return F.sigmoid(x)

class UNetBlock(nn.Module):
    def __init__(self, bn, in_, out):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, out, 3, padding=1)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1)
        self.bn = bn
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out)
            self.bn2 = nn.BatchNorm2d(out)
        self.activation = F.relu

    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.bn: x = self.bn2(x)
        x = self.activation(x)
        return x


class UNet_BN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.bn = hps.bn
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        if self.bn: self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        if self.bn: self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        if self.bn: self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        if self.bn: self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        if self.bn: self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        if self.bn: self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        if self.bn: self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        if self.bn: self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        if self.bn: self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        if self.bn: self.bn10 = nn.BatchNorm2d(512)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv11 = nn.Conv2d(512+256, 256, 3, padding=1)
        if self.bn: self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        if self.bn: self.bn12 = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256+128, 128, 3, padding=1)
        if self.bn: self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=1)
        if self.bn: self.bn14 = nn.BatchNorm2d(128)

        self.conv15 = nn.Conv2d(128+64, 64, 3, padding=1)
        if self.bn: self.bn15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)
        if self.bn: self.bn16 = nn.BatchNorm2d(64)

        self.conv17 = nn.Conv2d(64+32, 32, 3, padding=1)
        if self.bn: self.bn17 = nn.BatchNorm2d(32)
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1)
        if self.bn: self.bn18 = nn.BatchNorm2d(32)
        self.convfinal = nn.Conv2d(32, hps.n_classes, 3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.bn: x = self.bn2(x)
        skip0 = F.relu(x)


        skip1 = self.pool(skip0)
        skip1 = self.conv3(skip1)
        if self.bn: skip1 = self.bn3(skip1)
        skip1 = F.relu(skip1)
        skip1 = self.conv4(skip1)
        if self.bn: skip1 = self.bn4(skip1)
        skip1 = F.relu(skip1)

        skip2 = self.pool(skip1)
        skip2 = self.conv5(skip2)
        if self.bn: skip2 = self.bn5(skip2)
        skip2 = F.relu(skip2)
        skip2 = self.conv6(skip2)
        if self.bn: skip2 = self.bn6(skip2)
        skip2 = F.relu(skip2)

        skip3 = self.pool(skip2)
        skip3 = self.conv7(skip3)
        if self.bn: skip3 = self.bn7(skip3)
        skip3 = F.relu(skip3)
        skip3 = self.conv8(skip3)
        if self.bn: skip3 = self.bn8(skip3)
        skip3 = F.relu(skip3)

        skip4 = self.pool(skip3)
        skip4 = self.conv9(skip4)
        if self.bn: skip4 = self.bn9(skip4)
        skip4 = F.relu(skip4)
        skip4 = self.conv10(skip4)
        if self.bn: skip4 = self.bn10(skip4)
        skip4 = F.relu(skip4)

        x = self.upsample(skip4)
        x = torch.cat([x, skip3], 1)
        x = self.conv11(x)
        if self.bn: x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x)
        if self.bn: x = self.bn12(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, skip2], 1)
        x = self.conv13(x)
        if self.bn: x = self.bn13(x)
        x = F.relu(x)
        x = self.conv14(x)
        if self.bn: x = self.bn14(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, skip1], 1)
        x = self.conv15(x)
        if self.bn: x = self.bn15(x)
        x = F.relu(x)
        x = self.conv16(x)
        if self.bn: x = self.bn16(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, skip0], 1)
        x = self.conv17(x)
        if self.bn: x = self.bn17(x)
        x = F.relu(x)
        x = self.conv18(x)
        if self.bn: x = self.bn18(x)
        x = F.relu(x)
        x = self.convfinal(x)
        return F.sigmoid(x)

class UNet2(nn.Module):
    def __init__(self, nb_channels=12, n_classes=1, nb_filters=32, bn=False):
        super().__init__()
        # block = UNetBlock
        filter_factors = [1, 2, 4, 8, 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        filter_sizes = [nb_filters*j for j in filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            nf_in = nb_channels if i==0 else filter_sizes[i-1]
            self.down.append(UNetBlock(bn, nf_in, nf))
            if i!=0:
                self.up.append(UNetBlock(bn, nf_in+nf, nf_in))
        self.conv_final = nn.Conv2d(filter_sizes[0], n_classes, 1)


    def forward(self, x):
        xs = []
        # Encoder
        for i, down in enumerate(self.down):
            if i==0: x_in = x # starting input is x
            else: x_in = self.pool(xs[-1]) # else its the downsampled last convblock
            x_out = down(x_in) # apply a convblock
            xs.append(x_out)

        # Decoder
        x_out = xs[-1]
        # take the encoder output, upsample it, concatentate it with the encoder output of the block before that and
        # apply a convblock on them. Repeat until we are at the top layer again, then apply the last convlayer as
        # a classification layer
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            x_out = up(torch.cat([self.upsample(x_out), x_skip], 1))

        x_out = self.conv_final(x_out)
        return F.sigmoid(x_out)