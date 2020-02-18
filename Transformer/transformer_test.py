import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator # Fully Connected layer & Softmax after tgt_embed

		def forward(self, src, tgt, src_mask, tgt_mask): # Take in and process masked src and target sequences
			return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

		def encode(self, src, src_mask): # Encode part included src_embed and its mask
			return self.encoder(self.src_embed(src), src_mask)

		def decode(self, tgt, tgt_mask): # Output tgt_embed, its mask, and src_mask
			return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module): # Define standard linear + softmax
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N): # Produce N identical layers
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module): # Encoder inclded a stack of N layers
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N) # Clone layers
		self.norm = LayerNorm(layer.size) # Add LayerNorm, to prevent over-fitting and optimize computing
	
	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask) # Each input has its own mask
		return self.norm(x)

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e):-6
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features)) # Scaling
		self.b_2 = nn.Parameter(torch.zeros(features)) # Moving
		self.eps = eps # Minimum of denominator
	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	'''Residual Connection followed by LayerNorm'''
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Droupout(dropout) # Add droupout

	def forward(self, x, sublayer):
		'''Apply residual connection to sub layer with the same size'''
		return x + self.droupout(sublayer(self.norm)) # Implement residual connection

class EncoderLayer(nn.Module):
	'''Including self-attention and feed forward'''
	def __init__(self, size, self-attn, feed_forward, droupout):
		super(EncoderLayer, self).__init__()
		self.self-attn s= self.self-attn # Attention build later
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, droupout), 2) # Deep copy 2 connections, one for attention, the other one for simple NN
		self.size = size

	def forward(self, x, mask):
		'''Figure 1 (left) for connection'''
		x = self.sublayer[0](x, lambda x: self.self-attn(x, x, x, mask)) # The first sublayer for attn
		return self.sublayer[1](x, self.feed_forward) # The second sublayer for simple NN

	def attention(query, key, value, mask=None, droupout=None):
		'''Compute Scale Dot Product Attention'''
		d_k = query.size(-1) # size of query (token), default=64
		scores = torch.matmul(query, key.transpose(-2, -1) / math.stqr(d_k))
		if mask is not None: # Padding mask, as a tensor for padding for softmax to calculate
			scores = scores.masked_fill(mask == 0, -1e9)
		p_attn = F.softmax(scores, dim = -1)
		if droupout is not None: # Dropout, preventing from overfitting
			p_attn = droupout(p_attn)
		return torch.matmu(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, droupout=0.1):
		'''Take in model size and number of heads'''
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0 # Assure it's 0



