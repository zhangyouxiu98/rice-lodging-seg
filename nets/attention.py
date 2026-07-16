import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)

        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W — concatenate x and pattn1 along dimension 2.
                                                    # This is why pa2's input channels are 2 * dim:
                                                    # two tensors of dim channels each are stacked on a new dimension.
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2) # Use einops rearrange to merge dimensions,
                                                       # resulting in shape (B, C * 2, H, W)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class TBFE(nn.Module):
    def __init__(self, input_channels, reduction_N=32):
        super(TBFE, self).__init__()
        self.point_wise = nn.Conv2d(input_channels, reduction_N, kernel_size=1, padding=0, bias=False)
        self.depth_wise = nn.Sequential(nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(reduction_N), nn.ReLU(), )

        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 3), padding=(0, 0, 1),
                                stride=(1, 1, 1), bias=False)

        self.match_channels = nn.Conv2d(2 * reduction_N, input_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.point_wise(x) # (B, reduction_N, H, W)
        x_2 = self.depth_wise(x_1) # (B, reduction_N, H, W)
        x_2 = x_1 + x_2 # Residual connection to prevent vanishing gradients

        # DSC
        x_3 = x_1.unsqueeze(1)  # (B, 1, reduction_N, H, W)
        x_3 = self.conv3D(x_3)
        x_3 = x_3.squeeze(1) # (B, reduction_N, H, W)
        x = torch.cat((x_2, x_3), dim=1) # (B, 2 * reduction_N, H, W)
        x = self.match_channels(x)  # (B, input_channels, H, W)

        return x


class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        # return self.avgpool(x)
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=24):
        super().__init__()
        f = f
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1])
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g #(w + g) #self.gate(x, w)

if __name__ == '__main__':
        model = SpatialAttention()
        x = torch.randn(2, 3, 64, 64)
        x = model(x)[0]
        print(x.shape)
if __name__ == '__main__':

    input_tensor = torch.randn(1, 64, 32, 32)
    ca_module = ChannelAttention(dim=64)
    output = ca_module(input_tensor)
    print(f"Output channels: {output.shape}")

if __name__ == '__main__':

    input_x = torch.randn(1, 64, 32, 32)
    input_pattn1 = torch.randn(1, 64, 32, 32)
    pa_module = PixelAttention(dim=64)
    output = pa_module(input_x, input_pattn1)

    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    # Define input parameters
    batch_size = 4
    input_channels = 64
    height = 32
    width = 32

    # Create input tensor
    input_tensor = torch.randn(batch_size, input_channels, height, width)

    # Create TBFE instance
    tbfe = TBFE(input_channels=input_channels)

    # Forward pass
    output = tbfe(input_tensor)

    # Print output shape
    print(f"Output shape: {output.shape}")