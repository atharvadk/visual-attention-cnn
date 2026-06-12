import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        batch_size, channels, _, _ = x.size()

        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_out)

        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)

        out = out.view(batch_size, channels, 1, 1)

        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(out)

        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(
            in_channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.cbam = CBAM(out_channels)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.cbam(x)

        return x


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AttentionCNN, self).__init__()

        # Feature Extractor

        self.layer1 = CNNBlock(
            in_channels=3,
            out_channels=32,
            stride=1
        )

        self.layer2 = CNNBlock(
            in_channels=32,
            out_channels=64,
            stride=2
        )

        self.layer3 = CNNBlock(
            in_channels=64,
            out_channels=128,
            stride=2
        )

        self.layer4 = CNNBlock(
            in_channels=128,
            out_channels=256,
            stride=2
        )

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x