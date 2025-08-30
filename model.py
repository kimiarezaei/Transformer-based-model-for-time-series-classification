# define CNN model
import torch
import torch.nn as nn
from torchinfo import summary
import math

from utils import Params
 

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.TransformerEncoderLayer)):
        if hasattr(module,'weight'):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
        if hasattr(module,'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


# split each signal segment into tokens
class TokenEmbedding(nn.Module):
    def __init__(self, embedding_dim, kernel_size, stride):
        super().__init__()
        signal_length = 19200   # your signal length
        self.network = nn.Conv1d(1, embedding_dim, kernel_size=kernel_size, stride=stride),   # output_shape = (batch size, no. of channels, no. of tokens)
        self.num_patches = 1 + (signal_length - kernel_size) // stride

    def forward(self, X):
        out = self.network(X)
        return out.transpose(1, 2)        # Output shape: (batch size, no. of tokens, no. of channels)  


# define the vision transformer
class MyTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # tensors obtained from resized signal
        self.token_embedding = TokenEmbedding(params.embedding_dim, params.kernel_size, params.stride)           # output_size = (batch size, no. of tokens, no. of channels)
        
        # Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, params.embedding_dim) )        # cls token is introduced as a learnable parameter
        num_tokens = self.token_embedding.num_patches + 1  # Add the cls token

        # Positional embedding 
        pe = self.sinusoidal_embedding(num_tokens, params.embedding_dim)  
        self.register_buffer("pos_embedding", pe)  # Register as a fixed tensor

        self.dropout = nn.Dropout(params.drop_out_stoch)   # prevents the model from relying too much on individual tokens (patches extracted from ECG segment) and forces it to focus on the whole signal segment(4mins).
        
        # Encoder layer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=params.embedding_dim, nhead=params.num_heads, dropout=params.drop_out_att, dim_feedforward=int(params.embedding_dim*2), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=params.num_blks)
        
        # last MLP (head)
        self.head = nn.Sequential(nn.LayerNorm(params.embedding_dim), nn.Linear(params.embedding_dim, 2))    # classify to 2 classes
        
    def sinusoidal_embedding(self, num_tokens, embed_dim):
        # Implement sinusoidal positional encoding for dynamic shapes
        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_tokens, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)  
        return pe.unsqueeze(0)   # add batch dimension
    
    def forward(self,x):
        y_token_embedding = self.token_embedding(x)
        y_concat = torch.cat((self.cls_token.expand(y_token_embedding.shape[0], -1, -1), y_token_embedding), 1)   # change the token size to the size of the batches. Then add the token class to the begining of each batch
        y_dropout = self.dropout(y_concat + self.pos_embedding)
        y_encoder = self.encoder_blocks(y_dropout)
        return self.head(y_encoder[:,0])     # give the first token (class token) to the head for classification

