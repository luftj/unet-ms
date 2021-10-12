import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

def crop_tensor(tensor, target_tensor):
    # tensor is [bs,c,h,w]
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = (tensor_size - target_size) // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        num_channels = in_channels
        num_classes = out_channels
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = DoubleConv(num_channels,64)
        self.down_conv_2 = DoubleConv(64,128)
        self.down_conv_3 = DoubleConv(128,256)
        self.down_conv_4 = DoubleConv(256,512)
        self.down_conv_5 = DoubleConv(512,1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = DoubleConv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = DoubleConv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = DoubleConv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = DoubleConv(128, 64)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, image):
        # image is [bs,c,h,w]

        # encoder
        x1 = self.down_conv_1(image) #
        x = self.max_pool_2x2(x1)
        x2 = self.down_conv_2(x) #
        x = self.max_pool_2x2(x2)
        x3 = self.down_conv_3(x) #
        x = self.max_pool_2x2(x3)
        x4 = self.down_conv_4(x) #
        x = self.max_pool_2x2(x4)
        x = self.down_conv_5(x)

        # decoder
        x = self.up_trans_1(x)
        y = crop_tensor(x4,x)
        x = self.up_conv_1(torch.cat([y,x],1))
        
        x = self.up_trans_2(x)
        y = crop_tensor(x3,x)
        x = self.up_conv_2(torch.cat([y,x],1))

        x = self.up_trans_3(x)
        y = crop_tensor(x2,x)
        x = self.up_conv_3(torch.cat([y,x],1))

        x = self.up_trans_4(x)
        y = crop_tensor(x1,x)
        x = self.up_conv_4(torch.cat([y,x],1))

        x = self.output(x)
        return x

def test():
    x = torch.randn((1, 3, 572, 572))
    # x = torch.randn((3, 1, 572, 572))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    assert preds.shape == (1,1,388,388)

if __name__ == "__main__":
    test()