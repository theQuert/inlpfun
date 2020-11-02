import torch
import torch.nn.functional as F
# Build the lookup table (Embeddings)
# nn.Embeddings(num_embedding, embedding_dim)
# d_model = 512, haeds = 8, d_k = d_v = 512/8 = 64
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = Embedding(vocab, d_model)
		self.d_model = d_model
    
	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)

# Add Postion Encoding to "Embedding"
class PositionalEncoding(nn.Module)
	def __init__(self, d_model, dropout, max_len = 5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional emcodings in log space
		# PE(pos, 2*i) = sin(pos/(10000^(2*i/d_model)))
		# PE(pos, (2*i+1)) = cos(pos/(10000^(2*i/d_model)))
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1) # Expand dimension
		div_term = torch.exp(torch.arange(0, d_model, 2) * # Relative position 
								-(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term) # Select odd row
		pe[:, 1::2] = torch.cos(position * div_term) # Select even row
		pe.unsqueeze(0)
		self.register_buffer['pe', pe]	

		# Feed Forward : Residual Connection
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
		return self.dropout(x)

# Input Tensor (Word Embedding + Position Embedding = Word Representation) (batch_size, seq_len, d_model)
class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Process with masked src and tgt sequences"
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed, memory, src_maskl, tgt_mask)

class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
	"Produce N idendical layers."
	return nn.ModuleList([copy.deepcopy(module)] for _ in range(N))

'''
    Between sublayers, "Residual Connection" and "LayerNorm" are needed

'''
class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
        mean = x.mean(-1, keep_dim=True)
        std = x.std(-1, keep_dim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	'''
        LayerNorm + Dropout + Residual Connection
	'''
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = Laynorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		'''  
            LayerNorm(Residual Connection + Dropout)
            R -> D -> L ** LayerNorm(sublayer(x + self.norm(x)))
		'''
		return x + self.dropout(sublayer(self.norm(x)))

# Encoder
# Each Encoder Layer includes (Multi-Head Attention + LayerNorm + Dropout + Residual Connection + Feed Forward)
# To sum up -> Each Encoder layer includes (self_attn + feed_forward)

'''
	Core encoder is a stack of N layers
'''

class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"Pass the input through each layer in turn."
        ''' 
            Add sequence and masks to layers
        '''
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		# 2 sublayers
        '''
            Deep copy 2 SublayerConnection, 1) one for attention, 2) the other one for simple feed-forward NN
        '''
		self.size = size
		self.sublayer = clones(SublayerConnection(size, dropout), 2)

	'''
		Attention and Feed_forward
	'''
	'''
		In the beginning, each x is the representation of the sentence.
		Between EncoderLayers, each "x" is is the output of the previous layer.
	'''

	def forward(self, x, mask):
    '''
        Sublayers for Encoders: 1) For attention. 2) For Feed-Forward NN
    '''
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

	'''
		Each layer has two sub-layers. The first is a multi-head attention, and the second is a simple position-wise
		fully connected feed-forward network
	'''
	'''
		Sub-layers includes [Self-attention + LayerNorm + sublayer(dropout) + Residual Connection]
							[Feed Forward + LayerNorm + sublayer(dropout) + Residual Connection ]
	'''
	'''
		1. Multi-Head Attention / Self-Attention / Feed Forward 
		2. LayerNorm
		3. sublayer(dropout)
        4. Residual Connection
	'''
	'''
		size: d_model, we use 512
		self_attn: The instance of "MultiHeadAttention", sublayer[0]
		feed_forward: The instance of "PositionwiseFeedForward", sublayer[1]
		dropout: The dropout rate, nn.Dropout
	'''

# Decoder
	'''
		Decoder is composed of a stack of N = 6 identical layers
	'''
class  Decoder(nn.Module):
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers: # Add memory, src_mask, tgt_mask for self-attention and Encoder-Decoder Attention
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

	'''
		In addition to the two sub-layers in each encoder layer, the decoder
		inserts a third sub-layer, which performs multi-head attention over 
		the output of the encoder stack.

        *) LayerNorm(Residual Connecton + Dropout)
	'''
	
class DecoderLayer(nn.Module):
	'''
		Decoder includes self-attn, src-attn, and feed forward
	'''
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn # Self-Attention from Encoder
		self.src_attn = src_attn   # Attention for Decoder (Encoder-Decoder Attention)
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory
        # Masked Multi-Head Attention, to prevent the model pre-seen the words which need the prediction.
        # The "Masked" is used to prevent the model "seen the future", the "subsequent mask" is "Lower triangular
        # matrix"
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Encoder-Decoder Multi-head Attention
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # "Linear Transform" and "Softmax" are put behind sublayer[3], to predict the prob of tokens
		return self.sublayer[2](x, self.feed_forward)

	''' 
		Prevent positions from attending to subsequent positions
		"i" can only depend on the known outputs at positions less than "i"
	'''
	def subsequent_mask(size):
		attn_shape = (1, size, size)
        # "subsequent_mask" is used for prevent decoder "see the future", and when the masks are not empty, 
        # we change 0 to 1e-9, then the 1e-9 would be closed after passing through Softmax
		subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

# Attention

def attention(query, key, value, mask=None, dropout=None):
	''' 
		Compute "Scaled Dot Product Attention"
	'''
	d_k = query.size(-1)
	'''
		query * key.transpose(-2, -1):
			[nbatches, 8, L, 64] * [nbatches, 8, 64, L] = [nbatches, 8, L, L]
		Do softmax to scores
		Shape of "p_attn" is [nbatches, 8, L, L]
		Shape of "value" is [nbatches, 8, L, 64]
		Shape of matmul(p_attn, value) is [nbatches, 8, L, 64]	

		We have 8 heads done with different matmul -> Get different "representation subspace"
	'''
	scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
	if mask is not None:
    '''
        Mask should be 0 / 1
    '''
    # we change 0 to 1e-9, then the 1e-9 would be closed after passing through Softmax
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
	def __init__(self, h, d_model, dropout = 0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		'''
			We assume d_k always equal to d_v
		'''
		'''
			h = 8: We have 8 parallel attention layers, aka "heads"
			For each layer/head: We use d_k = d_v = d_model / h = 512 / 8 = 64
			dropout rate = 0.1
		'''
		
		self.h = h
		self.d_k = d_model // h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1) # Expand dimension
		nbatches = query.size(0)

		'''
			Line #120: 
				lambda x: self.self_attn(x, x, x, mask)

			"x" is the embedding of initialized sentence or previous layer output from EncoderLayer
			Shape of "query": [nbatches, L, d_model] = [nbatches, L, 512]
		'''
		'''
			0) Do linear transform to "query", "key", "value" -> Shape of them: [nbatches, L, 512]
			1) Use "view()" to reshape -> Shape of them: [nbatches, L, 8, 64], d_k = 512 / 8 = 64
			2) Use "transpose()" to swap dim_1 and dim_2 -> Shape of them: [nbatches, 8, L, 64]
		'''
		query, key, value = [l(x)view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
							for l, x in zip(self.linears, (query, key, value))]
		x, self_attn = attention(query, key, value, mask = mask, dropout = self.dropout)
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		return self.linear[-1](x)
		

		


'''
	query: [nbatches, L, 512]
	We have 8 heads: 
	512 / 8 = 64
	The shape of query, key, value are [nbatches, L, 8, 64]
	Then transpose(1, 2) -> [nbatches, 8, L, 64]
	The shapes of query, key, value are [nbatches, 8, L, 64]

	query * key.transpose(-1, -2)
	[nbatches, 8, L, 64] * [nbatches, 8, 64, L] = [nbatches, 8, L, L]
	p_attn = F.softmax(scores)
	The shape of p_attn = [nbatches, 8, L, L]
	The shape of value is [nbatches, 8, L, 64]

	[nbatches, L, 512] [nbatches, L, 8, 64] [nbatches, 8, L, 64]
	[nbatches, 8, L, 64] * [nbatches, 8, 64, L] = [nbatches, 8, L, L]
	[nbatches, 8, L, L] * [nbatches, 8, L, 64] = [nbatches, 8, L, 64]
'''
class PositionwiseFeedForward(nn.Module):
	'''
		Second sub-layer of Encoder
		Implement FFN equation
	'''	
	'''
		FFN(x) = max(0, w_1*x + b1)*w_2 + b_2

        The dimensions of Input tensor and Output tensor is d_model,
        The dimension of context tensor is d_ff, we let d_ff bigger than d_model, let FFN capture more data 
        from d_model.

        We use d_model = 512, d_ff = 512 * 4 = 2048
	'''
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(w_1(x))))

# Decoder
'''
	Sub-layers of Decoder:
		1) Masked Multi-Headed Attention
		2) Encoder-Decoder Attention
		3) LinearLayer, used to predict probabilities of words
'''
class DecoderLayer(nn.Module):
	'''
		Decoder includes self-attn, src_attn, and feed forward
	'''
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = SublayerConnection((size, dropout), 3)

	def forward(self, x, memory, src_mark, tgt_mask):

		'''
			"m" is the output from Encoder
			"x" is from the last DecoderLayer
			"Encoder-Decoder Attention" is composed of "m" and "x" from Encoder and previous DecoderLayer respectively
		'''
		'''
			In self-attention layer, all of the queries, keys, values come from the same place, 
			-> Each position in the encoder can attend to all positions in the previous layer of the encoder
		'''
		'''
			Self-attention layer in the decoder also allow each positon in the decoder to attend to all positions in the 
			decoder up and including that position.

			We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
			We implement this inside of scaled dot-product attention by masking out all values in the input of the softmax
			which correspond to illegal connections.
		'''
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.stc_attn(m, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)


class Generator(nn.Module):
	'''
		Define standard linear + softmax generation step
	'''
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.softmax(self.proj(x), dim=-1)

# Full Model
def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
	'''
		Produce model with hyperparameters
	'''
	c = copy.deepcopy
	attn = MultiHeadAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout) 
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
	nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
	nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
	Generator(d_model, tgt_vocab))

	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model

# The dimension changes between encoder and decoder

## 1. The inputs of decoder are known.
## 2. Decoder can generate all the outputs at one feed-forward operation.
## 3. The shape of outputs from decoder is [maxlen_tgt, d_model]
## 4. Multiply with the pre-softmax linear layer [d_voca, d_model] ##(Be treated as word embedding,
# weight can be shared with the input embedding)
## 5. We have the distribution:
	# P(d_voca, maxlen_tgt) = W(d_voca, d_model) * Xt(maxlen_tgt, d_model)

# Projection:
	# 1) d_k = d_v = d = d_model / h = 512 / 8 = 64
	# 2) Project original dimension d_model -> d / h
	# 3) If the sequence length are n * n, each neuron has n * n possible values in consideration
	# 4) After projection, num_params are 2* n * (d / h)  
	# 5) We need to approach 2 * n * (d / h) to (n * n), but 2nd/h << n^2, and it's prone to be when h is large.
	# 6) We have low-dimension bottleneck