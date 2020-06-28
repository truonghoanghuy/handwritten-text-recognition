from torch import nn, Tensor


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
