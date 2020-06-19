from torch import nn, Tensor


class CRNN(nn.Module):

    def __init__(self, cnn_out_size, num_channel, num_class, hidden_size, leaky_relu=False):
        super().__init__()
        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        dilation = [1, 1, 1, 1, 1, 1, 0]
        stride = [1, 1, 1, 1, 1, 1, 1]
        depth = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(index, batch_normalization=False):
            in_channels = num_channel if index == 0 else depth[index - 1]
            out_channels = depth[index]
            cnn.add_module(f'conv{index}', nn.Conv2d(in_channels, out_channels,
                                                     kernel_size[index], stride[index], dilation[index]))
            if batch_normalization:
                cnn.add_module(f'batchnorm{index}', nn.BatchNorm2d(out_channels))
            if leaky_relu:
                cnn.add_module(f'relu{index}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{index}', nn.ReLU(True))

        conv_relu(0)
        cnn.add_module(f'pooling{0}', nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module(f'pooling{1}', nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, batch_normalization=True)
        conv_relu(3)
        cnn.add_module(f'pooling{2}', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, batch_normalization=True)
        conv_relu(5)
        cnn.add_module(f'pooling{3}', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, batch_normalization=True)  # 512x1x16

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(cnn_out_size, hidden_size, num_class)
        self.softmax = nn.LogSoftmax()

    def forward(self, input: Tensor, len_label):
        # conv features
        conv = self.cnn(input.contiguous())
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c * h]
        # rnn features
        output = self.rnn(conv)

        return output


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, out_features):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_features)

    def forward(self, input: Tensor):
        recurrent, _ = self.rnn(input)
        seq_len, batch, h = recurrent.size()  # h = num_directions * hidden_size
        fc_in = recurrent.view(seq_len * batch, h)
        output = self.embedding(fc_in)
        output = output.view(seq_len, batch, -1)
        return output


def create_model(config):
    return CRNN(cnn_out_size=config['cnn_out_size'], num_channel=config['num_of_channels'],
                num_class=config['num_of_outputs'], hidden_size=512)