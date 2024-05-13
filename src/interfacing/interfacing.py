import torch
from chatbot import Chatbot
import argparse

def args():
    parser = argparse.ArgumentParser(description='Live Audio Chatbot')
    parser.add_argument('--message_rate', type=int, default=30, help='The rate at which new messages are sent')
    parser.add_argument('--prompt_format', type=str, default='llama', help='The format of the model [llama/mistral]')
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-hf', help='The base model')
    parser.add_argument('--trained_model', type=str, default='', help='The new model')

    return parser.parse_args()

# simple interaction with the chatbot
def interact(chatbot):
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        
        bot_response = chatbot.generate(user_input)
        print("Bot: ", bot_response)

def main():
    # get cli args
    args = args()

    # set the device for the chatbot
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # initialize the chatbot
    chatbot = Chatbot(args.base_model, args.trained_model, device=device, format=args.prompt_format)
    # interact with the chatbot
    interact(chatbot)

if __name__=="__main__":
    main()