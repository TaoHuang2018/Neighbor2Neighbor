import torch
import torch.nn as nn
import torch.nn.init as init

DEBUG = False

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                    m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UpsampleCat(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        # TypeError: conv_transpose2d(): argument 'output_padding' (position 6) must be tuple of ints, not tuple
        # self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 2, 2, 0, False)
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 2, 2, 0, 0)
        initialize_weights(self.deconv, 0.1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return torch.cat([x1, x2], dim=1)


def conv_func(x, conv, blindspot):
    size = conv.kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad = nn.ConstantPad2d(padding=(0, 0, ofs, 0), value=0)
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot):
    if blindspot:
        pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x


def rotate(x, angle):
    if angle == 0:
        return x
    elif angle == 90:
        return torch.rot90(x, k=1, dims=(3, 2))
    elif angle == 180:
        return torch.rot90(x, k=2, dims=(3, 2))
    elif angle == 270:
        return torch.rot90(x, k=3, dims=(3, 2))


class UNet(nn.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48,
                 blindspot=False,
                 zero_last=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.zero_last = zero_last
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Encoder part
        self.enc_conv0 = nn.Conv2d(self.in_nc, self.n_feature, 3, 1, 1)
        self.enc_conv1 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv0, 0.1)
        initialize_weights(self.enc_conv1, 0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv2, 0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv3, 0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv4 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv4, 0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv5, 0.1)
        self.pool5 = nn.MaxPool2d(2)

        self.enc_conv6 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv6, 0.1)

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv5b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv5a, 0.1)
        initialize_weights(self.dec_conv5b, 0.1)

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv4b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv4a, 0.1)
        initialize_weights(self.dec_conv4b, 0.1)

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv3b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv3a, 0.1)
        initialize_weights(self.dec_conv3b, 0.1)

        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv2b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv2a, 0.1)
        initialize_weights(self.dec_conv2b, 0.1)

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = nn.Conv2d(self.n_feature * 2 + self.in_nc, 96, 3, 1,
                                    1)
        initialize_weights(self.dec_conv1a, 0.1)
        self.dec_conv1b = nn.Conv2d(96, 96, 3, 1, 1)
        initialize_weights(self.dec_conv1b, 0.1)
        if blindspot:
            self.nin_a = nn.Conv2d(96 * 4, 96 * 4, 1, 1, 0)
            self.nin_b = nn.Conv2d(96 * 4, 96, 1, 1, 0)
        else:
            self.nin_a = nn.Conv2d(96, 96, 1, 1, 0)
            self.nin_b = nn.Conv2d(96, 96, 1, 1, 0)
        initialize_weights(self.nin_a, 0.1)
        initialize_weights(self.nin_b, 0.1)
        self.nin_c = nn.Conv2d(96, self.out_nc, 1, 1, 0)
        if not self.zero_last:
            initialize_weights(self.nin_c, 0.1)

    def forward(self, x):
        # Input stage
        blindspot = self.blindspot
        if blindspot:
            x = torch.cat([rotate(x, a) for a in [0, 90, 180, 270]], dim=0)
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0, blindspot))
        x = self.act(conv_func(x, self.enc_conv1, blindspot))
        x = pool_func(x, self.pool1, blindspot)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2, blindspot))
        x = pool_func(x, self.pool2, blindspot)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3, blindspot))
        x = pool_func(x, self.pool3, blindspot)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4, blindspot))
        x = pool_func(x, self.pool4, blindspot)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5, blindspot))
        x = pool_func(x, self.pool5, blindspot)

        x = self.act(conv_func(x, self.enc_conv6, blindspot))

        # Decoder part
        if DEBUG:
            print('x before up5', x.shape, 'pool4', pool4.shape)
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a, blindspot))
        x = self.act(conv_func(x, self.dec_conv5b, blindspot))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a, blindspot))
        x = self.act(conv_func(x, self.dec_conv4b, blindspot))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a, blindspot))
        x = self.act(conv_func(x, self.dec_conv3b, blindspot))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a, blindspot))
        x = self.act(conv_func(x, self.dec_conv2b, blindspot))

        x = self.up1(x, pool0)

        # Output stage
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
            x = pad(x[:, :, :-1, :])
            x = torch.split(x, split_size_or_sections=x.shape[0] // 4, dim=0)
            x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
            x = torch.cat(x, dim=1)
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        return x


if __name__ == "__main__":
    import numpy as np
    x = torch.from_numpy(np.zeros((10, 3, 32, 32), dtype=np.float32))
    print(x.shape)
    net = UNet(in_nc=3, out_nc=3, blindspot=False)
    y = net(x)
    print(y.shape)
