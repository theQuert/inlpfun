import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
# Scale -> Mask -> softmax -> Self-Attention
# Scale ensure imput being too max, which influence training
# Mask is to clear point the order
# Encoder section self-attention -> Put into decoder as "k" and "V", which connect encoder attention and decoder attention

class ScaledDotProductAttention(nn.Module):

	def __init__(self, attention_dropout=0.0):
		super(ScaledDotProductAttention, self).__init__()
		self.dropout = nn.Dropout(attention_dropout)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, q, k, v, scale=None, attn_mask=None):
		attention = torch.bmm(q, k.transpose(1, 2))
		if scale:
			attention = attention * scale
		if attn_mask:
			attention = attention.masked_filled(attn_mask, -np.inf)

		# softmax calculation	
		attention = self.softmax(attention)
		# Add dropout
		attention = self.dropout(attention)
		# Dot product with v
		# Context attention btwn sequences, calculating weight
		context = torch.bmm(attention, v)
		return context, attention

class MultiHeadAttention(nn.Module):
	def __init__(self, model_dim=512, num_heads=8, dropout=0.8):
		super(MultiHeadAttention, self).__init__()

		self.dim_per_head = model_dim // num_heads
		self.num_heads = num_heads
		self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
		self.linear.q = nn.Linear(model_dim, self.dim_per_head * num_heads)
		self.linear.v = nn.Linear(model_dim, self.dim_per_head * num_heads)
		
		self.dot_product_attention = ScaledDotProductAttention(dropout)
		self.linear_final = nn.Linear(model_dim, model_dim)
		self.dropout = nn.Dropout(dropout)

		# Layer norm after Multi-head attention
		self.layer_norm = nn.LayerNorm(model_dim)
	def forward(self, key, value, query, attn_mask=None):
		# Residual Connection
		residual = query
		dim_per_head = self.dim_per_head
		num_heads = self.num_heads
		batch_size = key.size(0)

		# Linear Projection
		key = self.linear_k(key)
		value = self.linear_v(value)
		query = self.linear_q(query)

		# split by heads
		key = key.view(batch_size * num_heads, -1, dim_per_head)
		value = value.view(batch_size * num_heads, -1, dim_per_head)
		query = query.view(batch_size * num_heads, -1, dim_per_head)

		if attn_mask:
			attn_mask = attn_mask.repeat(num_heads, 1, 1)

		# Scaled dot product attention
		scale = (key.size(-1)) ** -0.5
		context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

		# Concat heads
		context = context.view(batch_size, -1, dim_per_head * num_heads)
		# final linear projection
		output = self.liear_final(context)
		# dropout
		output = self.dropout(output)
		# add residual and norm layer
		output = self.leyer_norm(residual + output)

		return output, attention

	def padding_mask(seq_k, seq_q):
		# Shape of seq_k and seq_q are [B, L]
		len_q = seq_q.size(1)
		# 'PAD' is 0
		pad_mask = seq_k.eq(0)
		# shape [B, L_q, L_k]
		pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1) 
		return pad_mask

	def sequence_mask(seq):
		batch_size, seq_len = seq.size()
		mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.unit8), diagonal=1)
		mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # [B, L, L]
		return mask

class PotsitionEncoding(nn.Module):
	def __inti__(self, d_model, max_seq_len):
		''' Initialization:
		Args:
			d_model: Dimention of the model, default as 512
			max_seq_len: max length of the text sequence'''
		super(PotsitionEncoding, self).__init__()
		# Creating Position Emcoding (PE) Maxtrix from paper
		position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
		# Even use with sin, odd with cos converting
		position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
		position_encoding[:, 1::2] = np.sin(position_encoding[:, 1::2])

		pad_row = torch.zeros([1, d_model]) # Add position encoding for PAD
		position_encoding = torch.cat((pad_row, position_encoding)) # Concatnate pad and position_encoding

		self.position_encoding = nn.Embedding(max_seq_len +1, d_model) # Create matrix for nn.Embedding(x, y): x words, each one with y dimension
		self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad = False)

	def forward(self, input_len):
		'''
		Args:
		input_lens: A tensor (high dimension vector), shape as [BATCH_SIZE, 1], each tensor represent length of a text sequence
		
		Returns:
		Return the position encoding of the sequence, and aligned sequence
		'''
		# Find the max length of the sequence
		max_len = torch.max(input_len)
		tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
		# Align sequence and add 0 to later of the sequence
		# range start from 1 to skip PAD[0]
		input_pos = tensor(
			[list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
		return self.position_encoding(input_pos)

# Position-wise Feed-Forward network
class PositionWiseFeedForward(nn.Module):

	def __init__(self, model_dim = 512, ffn_dim = 2048, dropout = 0.0):
		super(PositionWiseFeedForward, self).__init__()
		self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
		self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(model_dim)

	def forward(self, x):
		output = x.transpose(1, 2)
		output = self.w2(F.relu(self.w1(output)))
		output = slf.dropout(output.transpose(1, 2))

		# Add Residual Layer and Norm Layer
		output = self.layer_norm(x + output)
		return output

# Encoder
class EncoderLayer(nn.Module):
	def __init__(self, model_dim = 512, num_heads = 8, ffn_dim = 2048, dropout = 0.0):
		super(EncoderLayer, self).__init__()
		self.atttention = MultiHeadAttention(model_dim, num_heads, dropout)
		self.feed_forward = PositionWiseFeedForward(model_dim, 	ffn_dim, dropout)

	def forward(self, inputs, attn_mask=None):
		# Self-Attention
		context, attention = self.attention(inputs, inputs, padding_mask)
		# Feed Forward Network
		output = self.feed_forward(context)

		return output, attention

class Encoder(nn.Module):
	'''Contains with multiple layers'''
	def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
		super(Encoder, self).__init__()

		self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

		self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
      def forward(self, inputs, inputs_len):
      	output = self.seq_embedding(inputs)
      	output += self.pos_embedding(inputs_len)

      	self_attention_mask = padding_mask(inputs, inputs)

      	attentions = []
      	for encoder in self.encoder_layers:
      		output, attention = encoder(output, self_attention_mask)
      		attentions.append(attention)

class DecoderLayer(nn.Module):
	def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout = 0.0):
		super(DecoderLayer, self).__init__()

		self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
		self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

	def forward(self,
        dec_inputs,
        enc_outputs,
        self_attn_mask=None,
        context_attn_mask=None):
		