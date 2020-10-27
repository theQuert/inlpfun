# Build the lookup table (Embeddings)
# nn.Embeddings(num_embedding, embedding_dim)
# d_model = 512
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
		# PE(pos, 2*i+1) = cos(pos/(10000^(2*i/d_model)))
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * 
								-(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe.unsqueeze(0)
		self.register_buffer['pe', pe]		

		# Feed Forward : Resicual Connection
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
		return self.dropout(x)
# Input Tensor (Word Embedding + Position Embedding = Word Representation) (batch_size, seq_len, d_model)
# d_model = 512
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
		return F.log_softmax(self.proj(x), dim=1)

def clones(module, N):
	"Produce N idendical layers."
	return nn.ModuleList([copy.deepcopy(module)] for _ in range(N))

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keep_dim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Encoder
# Each Encoder Layer includes (Multi-Head Attention + Residual Connection + LayerNorm + Feed Forward)
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
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)


class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		# 2 sublayers
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size
	'''
		Attention and Feed_forward
	'''
	'''
		In the beginning, each x is the representation of the sentence.
		Between EncoderLayers, each "x" is is the output of the previous layer.
	'''

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
	''' A residual connection followed by a layer norm.
	'''
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = Laynorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		''' x -> norm(x) -> sublayer -> dropout 
		'''
		return x + self.dropout(sublayer(self.norm(x)))

	'''
		Each layer has two sub-layers. The first is a multi-head attention, and the second is a simple position-wise
		fully connected feed-forward network
	'''
	'''
		Sub-layers includes [Multi-Head + Residual Connection + LayerNorm]
							[Feed Forward + Residual Connection]
	'''
	'''
		1. Multi-Head Attention / Feed Forward 
		2. Residual Connection
		3. LayerNorm
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
	def Decoder(nn.Module):
		def __init__(self, layer, N):
			super(Decoder, self).__init__()
			self.layers = clones(layer, N)
			self.norm = LayerNorm(layer.size)
		
		def forward(self, x, memory, src_mask, tgt_mask):
			for layer in self.layers:
				x = layer(x, memory, src_mask, tgt_mask)
			return self.norm(x)

	'''
		In addition to the two sub-layers in each encoder layer, the decoder
		inserts a third sub-layer, which performs multi-head attention over 
		the output of the encoder stack.

		Similar to the encoder, we also employ residual connection around each
		of the sub-layers, followed the layer norm.
	'''
	
class DecoderLayer(nn.Module):
	'''
		Decoder includes self-attn, src-attn, and feed forward
	'''
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

	''' 
		Prevent positions from attending to subsequent positions
		"i" can only depend on the known outputs at positions less than "i"
	'''
	def subsequent_mask(size):
		attn_shape = (1, size, size)
		subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

# Attention
class MultiHeadAttention(nn.Module):
	def __init__(self, h, d_model, dropout = 0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		'''
			We assume d_k always equal to d_v
		'''
		'''
			h = 8: We have 8 parallel attention layers / "heads"
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
			mask = mask.unsqueeze(1)
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
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn
		


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
		Implement FFN equation
	'''

























