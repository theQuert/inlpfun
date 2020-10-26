class EncoderDecoder(nn.Moddule):
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


# Encoder
def clones(module, N):
	"Produce N idendical layers."
	return nn.ModuleList([copy.deepcopy(module)] for _ in range(N))

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
class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		# 2 sublayers
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		x = self.sublayer[0]