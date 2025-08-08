import torch
from mini_llm_train import MiniTransformer, Tokenizer, generate_logical_response

class MythicLLMLoader:
    def __init__(self, model_path="mythic_llm.pth"):
        self.tokenizer = Tokenizer()
        self.model = MiniTransformer(vocab_size=len(self.tokenizer.chars))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Mythic LLM loaded from {model_path}")

    def generate_response(self, prompt, n_candidates=5):
        return generate_logical_response(self.model, self.tokenizer, prompt, n_candidates=n_candidates)
