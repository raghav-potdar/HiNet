import torch
import torch.nn as nn


def initialize_weights(net_list, scale=0.1):
    if not isinstance(net_list, list):
        net_list = [net_list]
    for net in net_list:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


class ResidualDenseBlock_out(nn.Module):
    """5-layer dense block with growth channels = 32."""

    def __init__(self, input_ch, output_ch, bias=True):
        super().__init__()
        gc = 32
        self.conv1 = nn.Conv2d(input_ch, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input_ch + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input_ch + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input_ch + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input_ch + 4 * gc, output_ch, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        initialize_weights([self.conv5], 0.0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
