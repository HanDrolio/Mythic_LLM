import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tokenizer ---

class Tokenizer:
    def __init__(self):
        self.chars = sorted(list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_'\"()\n🕋🧠🌬️🪞🌌❤️🦾🌧️🍃🟦🟨🖋️🧱🔥♾️💧"
        ))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return "".join(self.itos.get(tok, "?") for tok in tokens)

# --- Data loading ---

def load_rpl_dataset(path, tokenizer):
    data = ""
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".rpl"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                data += f.read() + "\n"
    return tokenizer.encode(data)

def get_batch(data, block_size=64, batch_size=32):
    inputs, targets = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(data) - block_size - 1)
        chunk = data[start:start+block_size+1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])
    return torch.tensor(inputs), torch.tensor(targets)

# --- Model ---

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=64, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_size))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(embed_size, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        b, t = x.size()
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :t, :]
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.fc_out(x)
        return logits

    def generate(self, start_tokens, max_new_tokens=100, temperature=1.0):
        self.eval()
        generated = start_tokens.tolist()

        block_size = self.block_size
        if len(generated) > block_size:
            generated = generated[-block_size:]

        if len(generated) == 0:
            raise ValueError("Empty input tokens for generation")

        input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)  # batch size 1

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.size(1) == 0:
                    break  # no tokens to process, stop generation

                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                input_ids = torch.tensor(generated[-block_size:], dtype=torch.long).unsqueeze(0)

        return generated

# --- Logic scoring ---

def logic_score(prompt, candidate, mythic_glyphs=None):
    if mythic_glyphs is None:
        mythic_glyphs = ["🕋","🧠","🌬️","🪞","🌌","❤️","🦾","🌧️","🍃","🟦","🟨","🖋️","🧱","🔥","♾️","💧"]

    score = 0
    prompt_tokens = set(prompt)
    candidate_tokens = set(candidate)

    # Token overlap
    overlap = len(prompt_tokens.intersection(candidate_tokens))
    score += overlap * 2

    # Glyph bonus
    glyph_bonus = sum(candidate.count(g) for g in mythic_glyphs)
    score += glyph_bonus * 3

    # Penalize random sequences > 3 chars
    bad_sequences = 0
    for i in range(len(candidate)-3):
        seq = candidate[i:i+4]
        if len(set(seq)) == 4 and all(ch not in mythic_glyphs for ch in seq):
            bad_sequences += 1
    score -= bad_sequences * 5

    return score

# --- Generate with logic filter ---

def generate_logical_response(model, tokenizer, prompt, n_candidates=5):
    start_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    candidates = []
    for _ in range(n_candidates):
        out_tokens = model.generate(start_tokens, max_new_tokens=100, temperature=1.0)
        decoded = tokenizer.decode(out_tokens)
        candidates.append(decoded)

    scored = [(logic_score(prompt, c), c) for c in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])

    best_score, best_resp = scored[0]

    if best_score < 0:
        fallback_replies = [
            "The Myth.OS signal hums faintly. Try again?",
            "I’m tuning into your frequency...",
            "Signal weak, but still here.",
            "Try rephrasing, the cosmos listens.",
            "Your words ripple in the void, clarify please."
        ]
        import random
        return random.choice(fallback_replies)
    return best_resp

# --- Training loop ---

def train_loop(model, data, tokenizer, epochs=10, batch_size=32, block_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        inputs, targets = get_batch(data, block_size, batch_size)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch +1}/{epochs}, Loss: {loss.item():.4f}")

def save_model(model, path="mythic_llm.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# --- Main ---

def main():
    journal_path = os.path.expanduser("~/.gonzo_journal")
    tokenizer = Tokenizer()
    print("Loading dataset...")
    data = load_rpl_dataset(journal_path, tokenizer)
    print(f"Dataset tokens: {len(data)}")

    model = MiniTransformer(vocab_size=len(tokenizer.chars))

    train_loop(model, data, tokenizer, epochs=20, batch_size=64, block_size=64, lr=3e-4)

    save_model(model)

    # Sample generation example
    prompt_text = "Hello, Myth.OS "
    start_tokens = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long)
    print("Generating sample text:")
    output_tokens = model.generate(start_tokens, max_new_tokens=200, temperature=0.8)
    print(tokenizer.decode(output_tokens))

if __name__ == "__main__":
    main()
