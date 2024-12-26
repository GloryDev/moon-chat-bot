from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from moon_chat_bot.true_synchronicity import true_random_multinomial
import numpy as np

class TrueRandomGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print("Initializing TrueRandomGenerator...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set different tokens for padding and EOS
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Ensure proper padding
        self.chat_history = []
        print("Initialization complete")
        
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.9):
        # Format prompt with chat history
        full_prompt = ""
        for i, message in enumerate(self.chat_history[-3:]):
            full_prompt += message + self.tokenizer.eos_token
        full_prompt += prompt + self.tokenizer.eos_token
        
        # Encode the prompt and create attention mask
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Prevent exceeding model's context window
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Generate new tokens
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,  # Added attention mask
                max_length=len(input_ids[0]) + max_length,
                min_length=len(input_ids[0]) + 10,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response
        response = generated_text[len(full_prompt.replace(self.tokenizer.eos_token, "")):]
        
        # Update chat history
        self.chat_history.append(prompt)
        self.chat_history.append(response)
        
        return response.strip()