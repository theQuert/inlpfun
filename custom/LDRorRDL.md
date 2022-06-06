### LayerNorm
### Dropout
### Residual Connection

### LDR


### RDL

### def
- PositionEncoding:
     Add Position Info, then adding "Dropout"

### clones
- deepcopy(module, N)


### classes

- class(Embedding) 
    - __init__
    - forward()

- PositionalEncoding
    - __init__
    - forward()


- EncoderDecoder
    - __init__
    - encode
    - decode

- Generator
    - __init__
    - forward

### def

- clones

### classes

- LayerNorm
    - __init__
    - forward

- SublayerConnnection
    - __init__
    - forward: ResidualConnection(Dropout(LayerNorm)) (LDR)

### Encoder - classes

 - Encoder
    - __init__
    - forward: Add masks to each layer, and LayerNorm (Masks + LayerNorm)

- EncoderLayer
    - __init__
    - forward: Build 2 sublayers with "SublayerConnection", including "Self-Attention" and "Feed-Forward"
    - Sublayer[1]: Feed-Forward NN : LDR

### Decoder - classes
- Decoder
    - __init__
    - forward; Add masks, memory, src_mask, tgt_mask to x in each layer

- DecoderLayer
    - __init__
    - forward: Build 3 sublayers with "SublayerConnection", including "Self-Attention", "Encoder-Decoder Attention"              , and "Feed Forward"

    - subsequent_mask: "Lower Triangular Matrix", used to prevent the model "see the future"     
         Then, we change the 0 to 1e-9, cause the 1e-9 would close to 0 after passing through softmax
 
### def - Attention
- attention

### classes - MultiHeadAttention
- __init__
- forward: Expand dimeension with "unsqueeze", the complex dimension multiplication

### classes - PositionwiseFeedForward
- __init__
- forward: The RNN Feed-Forward we talked about, by using "relu" as activation function and pass through each layer         : return self.w2(self.dropout(F.relu(w_1(x))))

### make_model
- EncoderDecoder(Encoder, Decoder)
- nn.Sequential(Embedding(src_vocab))
- nn.Sequential(Embedding(tgt_vocab))
- Generator()
- (d_vocab, max_tgt_len) = (d_vocab, d_model) * transpose(max_seq_len, d_model)
