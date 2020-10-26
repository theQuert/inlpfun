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
	'''
		Attention and Feed_forward
	'''

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)