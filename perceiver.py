import torch
import torch.nn as nn
from functools import wraps
import torch.nn.functional as F
from torchinfo import summary
#from cond_pixelcnn.utils.nn import NonLinear
from transformer import TransformerDecoderLayer, TransformerEncoderLayer

'''
    Implementation assisted by https://github.com/lucidrains/perceiver-pytorch and https://arxiv.org/pdf/2103.03206.pdf
'''

def exists(val):
    return val is not None

class PerceiverEncoderLayer(TransformerEncoderLayer):
    def forward(self, x: torch.Tensor, latent: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters:
        - x: Input sequence.
        - latent: Latent vectors.
        - mask: Masks out attention to values where mask is set to 1. Should be shape (query_len x key_len/value_len), \
            (batch_size x n_heads x query_len x key_len/value_len), or (batch_size x key_len/value_len)
        """
        residual = latent
        latent = self.norm1(latent) if self.pre_norm else latent
        latent = residual + self.dropout(self.attention(latent, x, x, mask))
        latent = self.norm1(latent) if not self.pre_norm else latent

        residual = latent
        latent = self.norm2(latent) if self.pre_norm else latent
        latent = residual + self.dropout(self.mlp(latent))
        latent = self.norm2(latent) if not self.pre_norm else latent
        return latent

class PerceiverBART(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_encoder_layers: int, n_decoder_layers: int,
                 latent_dim: int, n_latent_vecs: int,
                 dim_feedforward: int, weight_tie_layers: bool,
                 dropout: float, pre_norm: bool = False,
                 activation_fn: nn.Module = nn.ReLU,
                 norm_fn: nn.Module = nn.LayerNorm,
                 bias: bool = True, pad_token_idx: int = 0):
        super().__init__()
        # DECODER
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(TransformerDecoderLayer(d_model, n_heads, dim_feedforward,
                                                               dropout, pre_norm, activation_fn, 
                                                               norm_fn, bias, kdim=latent_dim, vdim=latent_dim))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)

        # ENCODER
        self.encoder_layers = nn.ModuleList()
        # WEIGHT TYING (all encoder blocks except first)
        first_encoder_layer = PerceiverEncoderLayer(latent_dim, n_heads, dim_feedforward, dropout,
                                                    pre_norm, activation_fn, norm_fn, bias, 
                                                    kdim=d_model, vdim=d_model)
        if weight_tie_layers and n_encoder_layers > 1:
            subsequent_encoder_layer = TransformerEncoderLayer(latent_dim, n_heads, dim_feedforward, dropout,
                                                               pre_norm, activation_fn, norm_fn, bias)
            for i in range(n_encoder_layers):
                not_first_iter = i > 0
                if not_first_iter:
                    self.encoder_layers.append(subsequent_encoder_layer)
                else:
                    self.encoder_layers.append(first_encoder_layer)
        else:
            self.encoder_layers.append(first_encoder_layer)
            for i in range(n_encoder_layers - 1):
                self.encoder_layers.append(TransformerEncoderLayer(latent_dim, n_heads, dim_feedforward, dropout,
                                                                   pre_norm, activation_fn, norm_fn, bias))
        
        # TODO: find out what num_latent_vecs is
        # LATENT
        self.latent = nn.Parameter(torch.randn(n_latent_vecs, latent_dim))

    def forward(self, x, mask=None):
        x_emb = self.embedding(x)

        batch_size, _, _ = x_emb.shape
        latent = self.latent.repeat(batch_size, 1, 1)

        memory = [x_emb, latent]
        for layer in self.encoder_layers:
            memory = [layer(*memory, mask=mask)]
        memory = memory[0]

        out = [memory, x_emb]
        for layer in self.decoder_layers:
            out = [memory, layer(*out)]
        out = out[1]

        return out


class PerceiverBARTVAE(PerceiverBART):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, num_decoder_layers: int, 
                 latent_dim: int, num_latent_vecs: int, 
                 dim_feedforward: int, weight_tie_layers: bool, 
                 padding_idx: int = 0):
        super().__init__(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, num_latent_vecs, dim_feedforward, weight_tie_layers, padding_idx)

        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        """
        From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py#L107

        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x_emb = self.embedding(x)

        batch_size, _, _ = x_emb.shape
        latent = self.latent.repeat(batch_size, 1, 1)

        for layer in self.encoder_layers:
            memory = layer(x_emb, latent)
        
        print('mem', memory.shape)

        mu = self.fc_mean(memory)
        log_var = self.fc_var(memory)
        z = self.reparameterize(mu, log_var)

        print('z', z.shape)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[-1])
        return self.decoder(tgt=x_emb, memory=z, tgt_mask=causal_mask), mu, log_var

    def loss_function(self, x, mu, log_var, y):
        recon_loss = F.cross_entropy(x, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
class PerceiverBARTAMIM(PerceiverBART):
    def loss_function(self, x, mu, log_var, y):
        # TODO: must implement log standard normal to get q(x)
        # TODO: must get marginal prior q(x) = log(p(x|z)) + log(p(z)) - log(q(z|x)) = log decoder + log prior - log encoder
        recon_loss = F.cross_entropy(x, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

if __name__ == "__main__":
    VOCAB = 30
    HIDDEN = 256
    NHEAD = 16
    LAYERS = 512
    BATCH = 10
    SEQLEN = 26

    model = PerceiverBART(VOCAB, HIDDEN, NHEAD, LAYERS, LAYERS, 64, 5, 1024, False, 0.1)
    summary(model)
    src = torch.randint(low=0, high=VOCAB, size=(BATCH, SEQLEN))
    print(model(src).shape)
