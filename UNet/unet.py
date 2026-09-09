#%%
import torch
import torch.nn as nn
#%%
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
       
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
#%%

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, option):
        super().__init__()
        
        downlist = [ nn.Conv2d(in_channels, in_channels, kernel_size=(3, 4), stride=2, padding=1), 
                     nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=3, padding=0),
                     nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=5, padding=0) ]

        down = downlist[option]

        model = [down] + [DoubleConv(in_channels, out_channels)]
        self.down_conv = nn.Sequential(*model)

    def forward(self, x):
        return self.down_conv(x) 
#%%

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=(3,4), stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
#%%

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.5, inplace=True) ) 

    def forward(self, x):
        return self.conv(x)
#%%

class UnetGenerator_c(nn.Module):
    """Create a Unet-based generator based on static data of coarse resolution"""

    def __init__(self, input_nc, output_nc, ngf = 16):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet.
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_c, self).__init__()

        # construct unet structure
        self.model = DoubleConv(input_nc, ngf)

        self.down1 = Down(ngf, ngf * 2, 0)         ### batch * nvar * 9 * 12
        self.down2 = Down(ngf * 2, ngf * 4, 0)        ### batch * nvar * 5 * 6
        self.down3 = Down(ngf * 4, ngf * 8, 0)        ### batch * nvar * 3 * 3

        self.up1 = Up(ngf * 8, ngf * 4)
        self.up2 = Up(ngf * 4, ngf * 2)
        self.up3 = Up(ngf * 2, ngf)
        self.outc = OutConv(ngf, output_nc)

    def forward(self, input):
        """Standard forward"""
        x1 = self.model(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        y = self.outc(x)

        return y
#%%
