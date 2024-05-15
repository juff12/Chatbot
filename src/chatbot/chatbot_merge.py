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

class MergedChatbot():
    def __init__(self, base_model, new_model, device='cuda:0', format='llama'):
        # ignore warnings
        logging.set_verbosity(logging.ERROR)
        print('loading chatbot')
        self.prompt_format = format
        print('device:', device)
        self.device_map = device
        print('loading base model')
        self.base_model = self.load_base(base_model)
        print('loading tokenizer')
        self.tokenizer = self.load_tokenizer(base_model)
        print('loading model')
        self.model = self.load_model(new_model)
        print('building pipeline')
        self.pipe = self.build_pipeline()

    def _create_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        return bnb_config

    def load_model(self, new_model):
        model = PeftModel.from_pretrained(self.base_model, new_model)
        model = model.merge_and_unload()
        return model

    def load_base(self, base_model):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=self._create_bnb_config(),
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(e)
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model
    
    def load_tokenizer(self, base_model):
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def build_pipeline(self):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        return pipe

    def generate(self, prompt):
        # format the prompt
        prompt_f = self.format_prompt(prompt)
        # generate the response
        # decent results
        # result = self.pipe(
        #     prompt,
        #     do_sample=True,
        #     max_new_tokens=40,
        #     temperature=0.5,
        #     top_k=0,
        # )
        
        result = self.pipe(
            prompt_f,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=10,
            max_new_tokens=4000,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
        )
        output = self.format_output(prompt_f, result[0]['generated_text'])
        if output != '':
            return output
        return self.generate(prompt)
    
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
            return '<s> ' + prompt
        # format the prompt for mistral instruct
        elif self.prompt_format == 'mistral':
            return f"<s>[INST] {prompt} [/INST]"