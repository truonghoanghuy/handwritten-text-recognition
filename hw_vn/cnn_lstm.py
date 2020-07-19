from torch import nn, Tensor

from hw_vn.cnn import CNN
from hw_vn.bi_lstm import BidirectionalLSTM


class CRNN(nn.Module):
    def __init__(self, cnn_out_size, num_channel, num_class, hidden_size, leaky_relu=False):
        super().__init__()

        self.cnn = CNN(num_channel, leaky_relu)
        self.rnn = BidirectionalLSTM(cnn_out_size, hidden_size, num_class)
        self.softmax = nn.LogSoftmax()

    def forward(self, input: Tensor, len_label=None):
        # conv features
        conv = self.cnn(input.contiguous())
        conv = conv.permute(2, 0, 1)  # [w, b, c * h]

        # rnn features
        output = self.rnn(conv)

        return output


def create_model(config):
    return CRNN(cnn_out_size=config['cnn_out_size'], num_channel=config['num_of_channels'],
                num_class=config['num_of_outputs'], hidden_size=512)
