import torch.nn as nn
from dataset import CSVDataset, SMILESDataLoader, concat_tensors, expand_tensor_dim, get_encoders, load_vocab_from_file, pad_to_len
from loss import recon_loss
from perceiver import PerceiverBART, PerceiverBARTVAE
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    n_epochs = 5

    lr = 5e-5

    d_model = 256
    n_heads = 4
    n_encoder_layers = 4
    n_decoder_layers = 4
    latent_dim = 32
    n_latent_vecs = 5
    dim_feedforward = 1024
    weight_tie_layers = True
    dropout = 0.1
    pre_norm = True
    activation_fn = nn.GELU
    norm_fn = nn.LayerNorm
    average_latent = False
    bias = True

    batch_size = 64

    pad_fn = lambda x: pad_to_len(x, [0, 0], batch_seq_idx=0, seq_dim=0)
    expand_fn = lambda x: expand_tensor_dim(x, [0, 1])
    concat_fn = lambda x: concat_tensors(x, [0, 1])

    vocab = load_vocab_from_file('E:/MOLMIM/data/selfies_vocab.txt')
    model_str2num, vocab_str2num = get_encoders(vocab, ['[PAD]', '[UNK]'])
    dataset = CSVDataset('E:/MOLMIM/data/allmolgen_255maxlen_cano.csv')
    dataloader = SMILESDataLoader(dataset, model_str2num, vocab_str2num, shuffle=True, use_selfies=True, batch_size=batch_size,
                                  mlm_prob_overall=0.0, collate_fns=[pad_fn, expand_fn, concat_fn])

    
    model = PerceiverBARTVAE(len(model_str2num) + len(vocab_str2num), d_model, n_heads, n_encoder_layers, n_decoder_layers,
                          latent_dim, n_latent_vecs, dim_feedforward, weight_tie_layers, dropout, pre_norm, activation_fn,
                          norm_fn, average_latent, bias)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        running_loss = 0.
        last_loss = 0.

        for i, batch in enumerate(dataloader):
            if len(batch) == 2:
                inputs, labels = batch
                weights = None
            else:
                inputs, labels, weights = batch
            
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = model.loss_function(outputs, labels, weights)
            print(loss)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 9
                print(f'batch {i + 1} loss: {last_loss}')
                running_loss = 0.
