from torch import nn, Tensor

from hw_vn.cnn import CNN
from hw_vn.bi_lstm import BidirectionalLSTM
from hw_vn.self_attention import SelfAttention


class CnnLstmAttention(nn.Module):
    def __init__(self, cnn_out_size, num_channel, num_class, hidden_size, leaky_relu=False):
        super().__init__()

        self.cnn = CNN(num_channel, leaky_relu)
        self.rnn = BidirectionalLSTM(cnn_out_size, hidden_size, num_class)
        self.self_attention = SelfAttention(num_class)
        self.softmax = nn.LogSoftmax()

    def forward(self, input: Tensor, labels_length: Tensor):
        # conv features
        conv = self.cnn(input.contiguous())

        # rnn features
        conv = conv.permute(2, 0, 1)  # [w, b, c * h]
        rnn = self.rnn(conv)  # [sequence length, batch, hidden states]

        # self attention
        in_attention = rnn.permute(1, 0, 2)  # [batch, sequence length, hidden states]
        output, _ = self.self_attention(in_attention, labels_length)

        output = output.permute(1, 0, 2)
        return output


def create_model(config):
    return CnnLstmAttention(cnn_out_size=config['cnn_out_size'], num_channel=config['num_of_channels'],
                            num_class=config['num_of_outputs'], hidden_size=512)
