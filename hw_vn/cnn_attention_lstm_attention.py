from torch import nn, Tensor

from hw_vn.cnn import CNN
from hw_vn.bi_lstm import BidirectionalLSTM
from hw_vn.self_attention import SelfAttention


class CnnAttentionLstmAttention(nn.Module):
    def __init__(self, cnn_out_size, num_channel, num_class, hidden_size, leaky_relu=False):
        super().__init__()

        self.cnn = CNN(num_channel, leaky_relu)
        self.self_attention_cnn = SelfAttention(cnn_out_size)
        self.rnn = BidirectionalLSTM(cnn_out_size, hidden_size, num_class)
        self.self_attention_lstm = SelfAttention(num_class)
        self.softmax = nn.LogSoftmax()

    def forward(self, input: Tensor, labels_length: Tensor):
        # conv features
        conv = self.cnn(input.contiguous())

        # self attention
        in_attention_cnn = conv.permute(0, 2, 1)
        out_attention_cnn, _ = self.self_attention_cnn(in_attention_cnn, labels_length)  # [b, w, c * h]

        # rnn features
        out_attention_cnn = out_attention_cnn.permute(1, 0, 2)  # [w, b, c * h]
        rnn = self.rnn(out_attention_cnn)  # [sequence length, batch, hidden states]

        # self attention
        in_attention_lstm = rnn.permute(1, 0, 2)  # [batch, sequence length, hidden states]
        output, _ = self.self_attention_lstm(in_attention_lstm, labels_length)

        output = output.permute(1, 0, 2)
        return output


def create_model(config):
    return CnnAttentionLstmAttention(cnn_out_size=config['cnn_out_size'], num_channel=config['num_of_channels'],
                                     num_class=config['num_of_outputs'], hidden_size=512)
