import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from utils import generate, output_formating
from peft import PeftModel

# Ignore warnings
logging.set_verbosity(logging.ERROR)

def chatbot_response(user_input, pipe):
    # generate the response
    response = generate(user_input, pipe)
    # format the response
    response = output_formating(response)
    return response

def loadchatbot(model_name, device_map, new_model):
    use_4bit = True # Activate 4-bit precision base model loading
    bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models
    bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
    use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # merge the pretrained model with the new model
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    # return the model pipeline
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

def chatbot(model_name, device_map, new_model):
    pipe = loadchatbot(model_name, device_map, new_model)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        
        bot_response = chatbot_response(user_input, pipe)
        print("Bot: ", bot_response)

def main():
    model_name = 'NousResearch/Meta-Llama-3-8B'
    device_map = 'cuda'
    new_model = ''
    chatbot(model_name, device_map, new_model)

if __name__=="__main__":
    main()