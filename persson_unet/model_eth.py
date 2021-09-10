import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

def crop_tensor(tensor, target_tensor):
    # tensor is [bs,c,h,w]
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = (tensor_size - target_size) // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

def crop_tensor_delta(tensor, delta):
    # tensor is [bs,c,h,w]
    tensor_size = tensor.size()[2]
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        up_channel = features[-1]
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(
                    up_channel, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            up_channel = feature

        self.bottleneck = DoubleConv(features[-1], features[-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.downs[0](x)
        # print(x.size())
        skip_connections.append(x)
        x = self.pool(x)
        x = self.downs[1](x)
        # print(x.size())
        skip_connections.append(x)
        x = self.pool(x)
        x = self.downs[2](x)
        # print(x.size())

        x = crop_tensor_delta(x,15)
        # print(x.size())
        x = self.bottleneck(x)
        # print(x.size())

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # print(x.size())
            skip_connection = skip_connections[idx//2]
            # print("skip",skip_connection.size())
            skip_connection = crop_tensor(skip_connection,x)
            # print("crop",skip_connection.size())


            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 3, 320, 320))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print("preds",preds.size())
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()