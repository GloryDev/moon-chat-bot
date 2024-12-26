from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from moon_chat_bot.true_multinomial import TrueRandomGenerator
import torch.nn.functional as F
from transformers import pipeline
from multiprocessing import freeze_support

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print("Initializing moon...")
        self.true_random = TrueRandomGenerator(model_name)
        print("Moon ready!")
        
    def generate_response(self, user_input, max_length=50, temperature=0.9):
        try:
            response = self.true_random.generate(
                prompt=user_input,
                max_length=max_length,
                temperature=temperature
            )
            return response if response else "I need to think about that..."
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "I encountered an error while trying to respond."

def main():
    print("Starting Moon initialization...")
    chatbot = Chatbot()
    
    print("\nMoon is ready! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        try:
            response = chatbot.generate_response(user_input)
            print(f"Moon: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    freeze_support()
    main()