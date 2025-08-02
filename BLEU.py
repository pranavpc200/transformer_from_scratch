# bleu_eval.py

import torch
from transformer import Transformer
from transformer_train import translate, src_tok, tgt_tok
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the dataset (limited to samples for evaluation)
def load_sample_dataset(path, max_lines=100):
    samples = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en, de = parts[0], parts[1]
                samples.append((en, de))
    return samples

# BLEU evaluator
def evaluate_bleu(dataset, model, src_tok, tgt_tok):
    smoothie = SmoothingFunction().method4
    scores = []
    model.eval()

    for en, de in dataset:
        ref = de.split()
        pred = translate(en, model, src_tok, tgt_tok).split()
        score = sentence_bleu([ref], pred, smoothing_function=smoothie)
        scores.append(score)

        print(f"\nEnglish:    {en}")
        print(f"Predicted:  {' '.join(pred)}")
        print(f"Reference:  {de}")
        print(f"BLEU score: {score:.4f}")

    return sum(scores) / len(scores)

# Load trained model
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

# Load evaluation data and run BLEU
sample_dataset = load_sample_dataset("C:/gpt_from_scratch/deu.txt", max_lines=10)
bleu = evaluate_bleu(sample_dataset, model, src_tok, tgt_tok)
print(f"\n\U0001F539 BLEU Score on evaluation set: {bleu:.4f}")
