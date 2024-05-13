import torch
import re
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)

class Chatbot():
    def __init__(self, base_model, new_model, device='cuda:0', format='llama'):
        # ignore warnings
        logging.set_verbosity(logging.ERROR)
    
        self.prompt_format = format
        self.device_map = device
        self.bnb_config = self._create_bnb_config()
        self.base_model = self.load_base(base_model)
        self.tokenizer = self.load_tokenizer(base_model)
        self.model = self.load_model(new_model)
        self.pipe = self.build_pipeline()

    def _create_bnb_config(self):
        bnb_config = BitsAndBytesConfig.from_pretrained(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        return bnb_config

    def load_model(self, new_model):
        model = PeftModel(
            base_model=self.base_model,
            new_model=new_model,
        )
        model = model.merge_and_unload()
        return model

    def load_base(self, base_model):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
        )
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
        return base_model
    
    def load_tokenizer(self, base_model):
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def build_pipeline(self):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_map,
        )
        return pipe

    def generate(self, prompt):
        # format the prompt
        prompt = self.format_prompt(prompt)
        # generate the response
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.8,
            top_p=0.8,
            top_k=10,
            max_new_tokens=64,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
        )
        return self.format_output(prompt, result[0]['generated_text'])
    
    def format_output(self, input, output):
        # remove the input from the output
        output = output.replace(input, '')
        # remove the punctuation that is unnecessary
        output = output.replace(':', ',').replace(';', ',')
        output = re.sub(r'[{}()\[\]]', '', output)
        return output.strip()

    def format_prompt(self, prompt):
        # format the prompt for llama chatbot
        if self.prompt_format == 'llama':
            return prompt
        # format the prompt for mistral instruct
        elif self.prompt_format == 'mistral':
            return f"<s>[INST] {prompt} [/INST]"