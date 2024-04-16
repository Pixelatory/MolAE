import torch.nn as nn
from dataset import CSVDataset, ConcatTensorCollator, ExpandTensorCollator, MaskedLanguageModellingCollator, PadToLenCollator, SMILESDataLoader, get_encoders, load_vocab_from_file
from models.perceiver import PerceiverBART, PerceiverBARTVAE
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

    vocab = load_vocab_from_file('data/selfies_vocab.txt')
    model_str2num, vocab_str2num = get_encoders(vocab, ['[PAD]', '[UNK]', '[MASK]', '[CLS]'])

    mlm_collate = MaskedLanguageModellingCollator(mask_token_encoding=model_str2num['[MASK]'], 
                                                  vocab_size=len(model_str2num) + len(vocab_str2num))
    pad_collate = PadToLenCollator([model_str2num['[PAD]'], model_str2num['[PAD]'], 0])
    expand_tensor_collate = ExpandTensorCollator()
    concat_collate = ConcatTensorCollator()

    dataset = CSVDataset('data/allmolgen_255maxlen_cano.csv')
    dataloader = SMILESDataLoader(dataset, model_str2num, vocab_str2num, shuffle=True, use_selfies=True, batch_size=batch_size,
                                  collate_fns=[mlm_collate, pad_collate, expand_tensor_collate, concat_collate])

    # TODO: Look into using memory transformer (REMTransformer?) for autoencoder
    
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
