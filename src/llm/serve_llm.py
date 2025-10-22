import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = os.environ.get('MODEL_DIR', 'models/lora-chatlegal')

class LLMService:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == '__main__':
    s = LLMService()
    print(s.generate('Explique brevemente o que é contrato de trabalho:'))
