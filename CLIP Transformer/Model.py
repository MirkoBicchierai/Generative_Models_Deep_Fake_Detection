import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, num_heads=8,
                 num_layers=4, max_seq_length=10):
        super(TransformerClassifier, self).__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length + 1)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.aux_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        batch_size = x.size(0)

        x = self.input_norm(x)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.positional_encoding(x)
        features = self.transformer_encoder(x)
        cls_feature = features[:, 0]

        seq_features = features[:, 1:]
        attn_weights = self.attention_pool(seq_features)
        context_vector = torch.sum(attn_weights * seq_features, dim=1)

        combined_features = cls_feature + context_vector

        logits = self.classifier(combined_features)

        aux_logits = self.aux_classifier(cls_feature)

        return logits, aux_logits