# train_transformer.py

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from transformer import Transformer

# -------------------- 1. LOAD YOUR OWN DATA --------------------
def load_custom_dataset(path, max_lines=20000):
    src_texts, tgt_texts = [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines: break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en, de = parts[0], parts[1]
                src_texts.append(en)
                tgt_texts.append(de)
    return src_texts, tgt_texts

src_texts, tgt_texts = load_custom_dataset("C:/gpt_from_scratch/deu.txt")
sample_dataset = list(zip(src_texts, tgt_texts))

# -------------------- 2. BUILD TOKENIZERS --------------------
src_tokenizer = ByteLevelBPETokenizer()
tgt_tokenizer = ByteLevelBPETokenizer()

src_tokenizer.train_from_iterator(src_texts, vocab_size=5000, min_frequency=2, special_tokens=["<pad>", "<sos>", "<eos>"])
tgt_tokenizer.train_from_iterator(tgt_texts, vocab_size=5000, min_frequency=2, special_tokens=["<pad>", "<sos>", "<eos>"])

src_tokenizer.save_model("C:/gpt_from_scratch/tokenizers", "src")
tgt_tokenizer.save_model("C:/gpt_from_scratch/tokenizers", "tgt")

from tokenizers.implementations import ByteLevelBPETokenizer as BPE
src_tokenizer = BPE("C:/gpt_from_scratch/tokenizers/src-vocab.json", "C:/gpt_from_scratch/tokenizers/src-merges.txt")
tgt_tokenizer = BPE("C:/gpt_from_scratch/tokenizers/tgt-vocab.json", "C:/gpt_from_scratch/tokenizers/tgt-merges.txt")

# -------------------- 3. WRAPPER --------------------
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
        return self.tokenizer.decode([i for i in ids if i not in (self.pad, self.sos)])

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

src_tok = TokenizerWrapper(src_tokenizer)
tgt_tok = TokenizerWrapper(tgt_tokenizer)

# -------------------- 4. DATASET --------------------
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=20):
        self.src = src_texts
        self.tgt = tgt_texts
        self.stok = src_tokenizer
        self.ttok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = torch.tensor(self.stok.encode(self.src[idx], self.max_len))
        tgt_ids = torch.tensor(self.ttok.encode(self.tgt[idx], self.max_len))
        return src_ids, tgt_ids

train_data = TranslationDataset(src_texts, tgt_texts, src_tok, tgt_tok)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# -------------------- 5. MODEL --------------------
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=src_tok.pad)

# -------------------- 6. INFERENCE --------------------
def translate(sentence, model, src_tok, tgt_tok, max_len=20):
    model.eval()
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

# -------------------- 7. TRAIN (protected by __main__) --------------------
def train():
    for epoch in range(25):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)

            logits = model(src, tgt_input)
            logits = logits.view(-1, logits.size(-1))

            loss = criterion(logits, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "C:/gpt_from_scratch/transformer_model_etog.pt")
    print("\nTranslate: 'really?'")
    print("→", translate("really?", model, src_tok, tgt_tok))
