import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,           # 🔥 deeper model
            batch_first=True,
            bidirectional=True,
            dropout=0.3            # 🔥 dropout between layers
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.dropout(hidden)
        out = self.fc(x)

        return out