import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
  # this block contains the token embeddings and the positional encodings and calls the rest of the classes

  def __init__(self, vocab_size, context_length, model_dim, num_blocks, num_heads, hidden_dim):
    super().__init__()
    torch.manual_seed(0)
    # first is the token embeddings
    self.token_embed = nn.Embedding(vocab_size, model_dim)
    self.pos_embed = nn.Embedding(context_length, model_dim)
    self.blocks = nn.ModuleList()
    self.dropout = nn.Dropout(0.1)  # to avoid overfitting(optional adjustment)

    for _ in range(num_blocks):
      self.blocks.append(EncoderTransformerBlock(model_dim, num_heads, hidden_dim))


  def forward(self, x):
    token_embeddings = self.token_embed(x)
    B, T, D = token_embeddings.shape
    pos_ids = torch.arange(T,  device=x.device)
    pos_encodings = self.pos_embed(pos_ids)[None, :, :]  #this will return shape 1 ,T, D
    tot_embs = token_embeddings + pos_encodings
    encoder_output = self.dropout(tot_embs)
    for block in self.blocks:
      encoder_output = block(encoder_output)   # x -> (B, T, D)

    return encoder_output



  
# everything in encoder except embeddings
class EncoderTransformerBlock(nn.Module):
  def __init__(self, model_dim, num_heads, hidden_dim):
    super().__init__()
    

    self.mha = MultiHeadAttention(model_dim, num_heads)
    self.normlayer1 = nn.LayerNorm(model_dim)
    self.normlayer2 = nn.LayerNorm(model_dim)
    self.feedforward = nn.Sequential(
      nn.Linear(model_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, model_dim)
    )
    self.dropout = nn.Dropout(0.1)  # to avoid overfitting(optional adjustment)

  def forward(self, x):
    o1 = self.normlayer1(x + self.mha(x))
    o2 = self.normlayer2(o1 + self.feedforward(o1))

    return self.dropout(o2)



class MultiHeadAttention(nn.Module):
 # num_heads for the number of heads needed for parallel attention
  def __init__(self, embedding_dim, num_heads):
    super().__init__()

    # now we wrap this class around self attention
    self.heads = nn.ModuleList()
    for _ in range(num_heads):
      self.heads.append(SingleHeadAttention(embedding_dim, embedding_dim//num_heads))

    self.proj_out = nn.Linear(embedding_dim, embedding_dim) # this is to finally project all the head output into one single linear projection of dimension embedding_dim itself after concatenation

  def forward(self,x):
    output = []

    for head in self.heads:
      output.append(head(x))

    # now concatenate all heads
    con_cated = torch.concat(output, dim = -1)

    output = self.proj_out(con_cated)

    return output


class SingleHeadAttention(nn.Module):
  def __init__(self, embedding_dim, attention_dim):
    super().__init__()

    self.query = nn.Linear(embedding_dim, attention_dim)
    self.key = nn.Linear(embedding_dim, attention_dim)
    self.value = nn.Linear(embedding_dim, attention_dim)

    self.attention = ScaledDotProductAttention()


  def forward(self, x):
    Q = self.query(x)
    K = self.key(x)  # this K used as input to cross attention
    V = self.value(x) # this V used as input to cross attention

    output, attention_weights = self.attention(Q, K, V)

    return output

class ScaledDotProductAttention(nn.Module):

  def __init__(self):
    super().__init__()


  def forward(self, Q, K, V):

    #implementing self attention

    # attention( Q, K, V) = softmax( Q . K^T // root(dk)) . V

    scores = Q @ torch.transpose(K, 1, 2)
    B, T, A = K.shape

    scores = scores / (A**0.5)

    scores = nn.functional.softmax(scores, dim = -1)

    output = scores @ V  # shape of output -> (B, T, A)

    return output, scores
  
  # class having both encoder and decoder

class Transformer(nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, context_length, model_dim, num_blocks, num_heads, hidden_dim):
    super().__init__()
    self.encoder = EncoderBlock(src_vocab_size, context_length, model_dim, num_blocks, num_heads, hidden_dim)
    self.decoder = DecoderBlock(tgt_vocab_size, context_length, model_dim, num_blocks, num_heads, hidden_dim)
    self.lm_head = nn.Linear(model_dim, tgt_vocab_size)

  def forward(self, src, tgt):
    enc_out = self.encoder(src)          # (B, T_enc, D)
    dec_out = self.decoder(tgt, enc_out) # (B, T_dec, D)
    logits = self.lm_head(dec_out)       # (B, T_dec, V)
    return logits


# everything in decoder

class DecoderBlock(nn.Module):
  def __init__(self, vocab_size, context_length, model_dim, num_blocks, num_heads, hidden_dim):
    super().__init__()
    torch.manual_seed(0)
    # first is the token embeddings
    self.token_embed = nn.Embedding(vocab_size, model_dim)
    self.pos_embed = nn.Embedding(context_length, model_dim)
    self.blocks = nn.ModuleList()
    self.dropout = nn.Dropout(0.1)  # to avoid overfitting(optional adjustment)

    for _ in range(num_blocks):
      self.blocks.append(DecoderTransformerBlock(model_dim, num_heads, hidden_dim))


  def forward(self, x, encoder_output):
    token_embeddings = self.token_embed(x)
    B, T, D = token_embeddings.shape
    pos_ids = torch.arange(T,  device=x.device)
    pos_encodings = self.pos_embed(pos_ids)[None, :, :]  #this will return shape 1 ,T, D
    tot_embs = token_embeddings + pos_encodings
    decoder_output = self.dropout(tot_embs)
    for block in self.blocks:
      decoder_output = block(decoder_output, encoder_output)   # x -> (B, T, D)

    return decoder_output


class DecoderTransformerBlock(nn.Module):
  # adds the feed forward
  def __init__(self, model_dim, num_heads, hidden_dim):
    super().__init__()
    self.masked_mha = MaskedMHSA(model_dim, num_heads)
    self.cross_attn = CrossAttention(model_dim)
    self.normlayer1 = nn.LayerNorm(model_dim)
    self.normlayer2 = nn.LayerNorm(model_dim)
    self.feedforward = nn.Sequential(
      nn.Linear(model_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, model_dim)
      )
    self.dropout = nn.Dropout(0.1)

  def forward(self,x, encoder_output):
    x1 = self.masked_mha(x)
    x2 = self.cross_attn(encoder_output, x1)
    x3 = self.normlayer1(x1 + x2)
    x4 = self.normlayer2(x3 + self.feedforward(x3))
    return x4


class CrossAttention(nn.Module):
  # this class implements cross attention where it takes Q from the decoder with K,V from the encoder as inputs to predict the next token
  def __init__(self, embedding_dim):
    super().__init__()
    self.query = nn.Linear(embedding_dim, embedding_dim)
    self.key = nn.Linear(embedding_dim, embedding_dim)
    self.value = nn.Linear(embedding_dim, embedding_dim)
    self.attention = ScaledDotProductAttention()
    self.proj_out = nn.Linear(embedding_dim, embedding_dim)
    self.dropout = nn.Dropout(0.1)
    # will also add norm layer here
    self.normlayer = nn.LayerNorm(embedding_dim)

  def forward(self, encoder_output, decoder_hidden):
    # these are inputs to this attention block

    Q = self.query(decoder_hidden) # decoder_hidden has shape: (B, T_dec, D) where T_dec: no of tokens generated in the output sequence -> T_dec grows as the no of output sequence generated increases with time

    K = self.key(encoder_output)    # encoder_output has shape: (B, T_enc, D) where T_enc: no of tokens generated in the input sequence -> this is fixed as the full sentence is given in the beginning to translate( in our case)
    V = self.value(encoder_output)
    attn_wts, _ = self.attention(Q, K, V)
    proj = self.proj_out(attn_wts)
    cross_out = decoder_hidden + self.dropout(proj)

    return self.normlayer(cross_out)  # shape: (B, T, D)


class MaskedMHSA(nn.Module):
  def __init__(self, embedding_dim, num_heads):
    super().__init__()

    self.heads = nn.ModuleList()
    self.proj_out = nn.Linear(embedding_dim, embedding_dim)
    # will also add norm layer here
    self.normlayer = nn.LayerNorm(embedding_dim)
    for _ in range(num_heads):
      self.heads.append(MaskedSelfAttention(embedding_dim, embedding_dim // num_heads))



  def forward(self,x):
    output = []
    for head in self.heads:
      output.append(head(x))  #output -> (B, T, A) where A is the attention_dim for each head

    con_cated = torch.concat(output, dim=-1)  
    proj = self.proj_out(con_cated) # (B, T, D) where D: embedding_dim
    decoder_output = x + self.normlayer(proj)
    return decoder_output  # now this goes into cross attention block
  

class MaskedSelfAttention(nn.Module):
  def __init__(self, embedding_dim, attention_dim):
    super().__init__()

    self.query = nn.Linear(embedding_dim, attention_dim)
    self.key = nn.Linear(embedding_dim, attention_dim)
    self.value = nn.Linear(embedding_dim, attention_dim)

    self.attention = MaskedScaledDotProductAttention()


  def forward(self, x):
    Q = self.query(x)  # this Q is used as input in cross attention
    K = self.key(x)
    V = self.value(x)

    output, attention_wts = self.attention(Q, K, V) # shape of output -> (B, T, A)

    return output

    

class MaskedScaledDotProductAttention(nn.Module):
  def __init__(self):
    super().__init__()



  def forward(self, Q, K, V):

    #implementing self attention

    # attention( Q, K, V) = softmax( Q . K^T // root(dk)) . V

    scores = Q @ torch.transpose(K, 1, 2)
    A = K.size(-1)

    scores = scores / (A**0.5)

    T = Q.size(1)
    # for masking the future tokens
    mask = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))

    scores = nn.functional.softmax(scores, dim = -1)

    output = torch.matmul(scores, V)

    return output, scores


