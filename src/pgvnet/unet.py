import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv1(x) + self.conv3(x) + self.conv5(x)


def conv_block(in_channels, out_channels, norm_groups=4):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(norm_groups, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(norm_groups, out_channels),
        nn.ReLU(inplace=True),
    )


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=32):
        super(AttentionUNet, self).__init__()
        self.enc1 = conv_block(in_channels, features)
        self.sa1 = SpatialAttention()
        self.ms1 = MultiScaleFusion(features)

        self.enc2 = conv_block(features, features * 2)
        self.sa2 = SpatialAttention()
        self.ms2 = MultiScaleFusion(features * 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(features * 2, features * 4)
        self.sa_b = SpatialAttention()
        self.ms_b = MultiScaleFusion(features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1 = self.ms1(self.sa1(enc1))
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.ms2(self.sa2(enc2))

        bottleneck = self.bottleneck(self.pool(enc2))
        bottleneck = self.ms_b(self.sa_b(bottleneck))

        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final_conv(dec1)


if __name__ == "__main__":
    
    input_tensor = torch.randn(1, 22, 4, 4) # Batch size, channels, height, width   
    model        = AttentionUNet(in_channels=22, out_channels=64)
    output       = model(input_tensor)
    print("Output shape:", output.shape)



