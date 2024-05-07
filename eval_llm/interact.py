from pathlib import Path
from dataclasses import dataclass

import torch
import simple_parsing
from transformers import pipeline
from utils import load_model_and_tokenizer

@dataclass
class Args:
    model_id: str = 'capecape/otto_bis/vre1cdrm-mistral:v0'
    temperature: float = 0.7
    max_new_tokens: int = 128
    prompt_file: Path = Path("prompts/mistral_simple.txt")

args = simple_parsing.parse(Args)

print("Welcome to the interactive chat experience!\nType 'quit' to exit the chat or 'new' to start a new conversation.")

print(f"Loading model from {args.model_id}")
model, tokenizer = load_model_and_tokenizer(args.model_id)

system_prompt = args.prompt_file.read_text()

while True:
    user_input = input("\033[94mUser: \033[0m")
    if user_input.lower() == "quit":
        break
    user_input = system_prompt.format(user=user_input, answer="")
    print("===============================================================")
    print(f"User\n{user_input}")
    print("===============================================================")
    encoded_input = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
    print(f"Encoded input\n{encoded_input}")
    print("===============================================================")
    out = model.generate(
        encoded_input, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature)
    print("===============================================================")
    generated_text = tokenizer.decode(out[0][len(encoded_input[0]):], skip_special_tokens=True)
    print(f"Answer\n{generated_text}")
    print("===============================================================")

print("Thank you for chatting. Goodbye!")
