# test_model.py

import torch
from transformer import Transformer
from tokenizers.implementations import ByteLevelBPETokenizer as BPE

# -------------------- 1. Load Tokenizers --------------------
src_tokenizer = BPE("C:/gpt_from_scratch/tokenizers/src-vocab.json", "C:/gpt_from_scratch/tokenizers/src-merges.txt")
tgt_tokenizer = BPE("C:/gpt_from_scratch/tokenizers/tgt-vocab.json", "C:/gpt_from_scratch/tokenizers/tgt-merges.txt")

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.token_to_id("<pad>")
        self.sos = self.tokenizer.token_to_id("<sos>")
        self.eos = self.tokenizer.token_to_id("<eos>")

    def encode(self, text, max_len):
        ids = self.tokenizer.encode(text).ids
        ids = [self.sos] + ids[:max_len - 2] + [self.eos]
        ids += [self.pad] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        if self.eos in ids:
            ids = ids[:ids.index(self.eos)]
        return self.tokenizer.decode([i for i in ids if i != self.pad and i != self.sos])

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

src_tok = TokenizerWrapper(src_tokenizer)
tgt_tok = TokenizerWrapper(tgt_tokenizer)

# -------------------- 2. Load Trained Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_vocab_size=src_tok.vocab_size(),
    tgt_vocab_size=tgt_tok.vocab_size(),
    context_length=20,
    model_dim=64,
    num_blocks=2,
    num_heads=4,
    hidden_dim=128
).to(device)

model.load_state_dict(torch.load("C:/gpt_from_scratch/transformer_model_etog.pt", map_location=device))
model.eval()

# -------------------- 3. Translate Function --------------------
def translate(sentence, model, src_tok, tgt_tok, max_len=20):
    src_ids = torch.tensor(src_tok.encode(sentence, max_len)).unsqueeze(0).to(device)
    tgt_ids = torch.tensor([tgt_tok.sos]).unsqueeze(0).to(device)

    for _ in range(max_len):
        logits = model(src_ids, tgt_ids)
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        if next_token.item() == tgt_tok.eos:
            break

    output = tgt_tok.decode(tgt_ids[0].tolist())
    return output

# -------------------- 4. Run Translation --------------------
test_sentence = "Get Tom Go away!"
print("EN:", test_sentence)
print("→ DE:", translate(test_sentence, model, src_tok, tgt_tok))
