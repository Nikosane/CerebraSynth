import torch
import torch.nn as nn
import torch.optim as optim

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=32, num_heads=4, num_layers=2):
        super(TransformerAutoencoder, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        self.latent_projection = nn.Linear(input_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent_projection(encoded)
        decoded = self.decoder(self.output_projection(latent), encoded)
        return decoded
    
    def encode(self, x):
        return self.latent_projection(self.encoder(x))
    
    def decode(self, x, memory):
        return self.decoder(self.output_projection(x), memory)
