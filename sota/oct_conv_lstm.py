# encoding: utf-8
import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], hx], dim=1) # (B, C, H, W)                           
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

class Encoder(nn.Module):
    def __init__(self, cfg_encoder):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(cfg_encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, cfg_decoder):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(cfg_decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == 'sigmoid': layers.append(nn.Sigmoid())
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer:
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                x = getattr(self, layer)(x)
                encoder_outputs[idx] = x
        return x

#https://github.com/czifan/ConvLSTM.pytorch
class ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
        cfg_encoder = [('conv', 'leaky', 2, 32, 3, 1, 2),
                        ('convlstm', '', 32, 32, 3, 1, 1),
                        ('conv', 'leaky', 32, 64, 3, 1, 2),
                        ('convlstm', '', 64, 64, 3, 1, 1),
                        ('conv', 'leaky', 64, 128, 3, 1, 2),
                        ('convlstm', '', 128, 128, 3, 1, 1)]
        cfg_decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
                    ('convlstm', '', 128, 64, 3, 1, 1),
                    ('deconv', 'leaky', 64, 32, 4, 1, 2),
                    ('convlstm', '', 64, 32, 3, 1, 1),
                    ('deconv', 'leaky', 32, 32, 4, 1, 2),
                    ('convlstm', '', 34, 32, 3, 1, 1), #in_ch - out_ch = channels
                    ('conv', 'sigmoid', 32, 2, 1, 0, 1)]
        self.encoder = Encoder(cfg_encoder)
        self.decoder = Decoder(cfg_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    scan =  torch.rand(5, 100, 2, 64, 64).cuda()
    model = ConvLSTM().cuda()
    out =  model(scan)
    print(out.shape)