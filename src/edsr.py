import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic residual block with two 3x3 convolutions and optional residual scaling.

    Args:
        channels (int): Number of input and output channels.
        res_scale (float): Scaling factor for residual connection.
    """
    def __init__(self, channels, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale
        out += residual
        return out

class EDSREncoder(nn.Module):
    """
    EDSR-inspired encoder using residual blocks for feature extraction.

    Args:
        in_channels (int): Number of input channels.
        num_features (int): Number of intermediate features.
        num_blocks (int): Number of residual blocks.
        out_channels (int): Number of output (latent) channels.
        res_scale (float): Residual scaling factor within blocks.
    """
    def __init__(self, in_channels=2, num_features=64, num_blocks=16, out_channels=128, res_scale=0.1):
        super(EDSREncoder, self).__init__()

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)]
        )

        self.tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.latent_proj = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.tail(x)
        x += residual
        x = self.latent_proj(x)
        return x

if __name__ == "__main__":
    model = EDSREncoder(in_channels=2, num_features=64, num_blocks=16, out_channels=128)
    input_tensor = torch.randn(1, 2, 64, 64)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
