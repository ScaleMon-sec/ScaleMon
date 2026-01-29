import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        latent = self.fc(h[-1])

        batch_size = x.size(0)
        seq_len = x.size(1)

        decoder_input = torch.zeros(batch_size, seq_len, x.size(2)).to(x.device)

        decoder_output, _ = self.decoder_lstm(decoder_input)
        output = self.decoder_fc(decoder_output)
        return output

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=32, hidden_dim=64, num_layers=2, nhead=4, seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        B, T, F = x.shape

        x_embed = self.embedding(x) + self.pos_embedding[:, :T]
        memory = self.encoder(x_embed)
 
        latent = self.to_latent(memory[:, -1]) 

        dec_input = torch.zeros(B, self.seq_len, self.input_dim, device=x.device)
        dec_embed = self.embedding(dec_input) + self.pos_embedding[:, :self.seq_len]

        latent_expand = self.from_latent(latent).unsqueeze(1)
        latent_memory = latent_expand.repeat(1, self.seq_len, 1)

        dec_output = self.decoder(dec_embed, latent_memory)

        return self.output_layer(dec_output)