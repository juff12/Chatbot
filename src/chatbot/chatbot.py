import torch
import re
from peft import PeftModel
import logging as logg
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from abc import ABC, abstractmethod

class AbstractChatbot(ABC):
    def __init__(self, model_name, tokenizer_name, device, format):
        super().__init__()
        self._logger = self._init_logger()
        self.prompt_format = self._get_format(format)
        self.device_map = device
        self.tokenizer = self.load_tokenizer(tokenizer_name)
        self.model = self.load_model(model_name)
        self.pipe = self.build_pipeline()
    
    @abstractmethod
    def _get_format(self, format):
        if format == 'llama':
            return '<s> {}'
        elif format == 'mistral':
            return '<s>[INST] {} [/INST'
        elif format == 'custom':
            return format
        # assume no format is given
        return '{}'

    @abstractmethod
    def _init_logger(self):
        """
        Initialize the logger for logging messages.
        """
        logger = logg.getLogger(__name__)
        handler = logg.StreamHandler()
        formatter = logg.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logg.DEBUG)
        logging.set_verbosity(logging.ERROR)
        return logger

    @abstractmethod
    def _create_bnb_config(self):
        self._logger.info("('>>> Creating BitsAndBytesConfig object")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self._logger.info("BitsAndBytesConfig object created <<<")
        return bnb_config
    
    @abstractmethod
    def load_model(self, model_name):
        self._logger.info(f">>> Loading model from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self._create_bnb_config(),
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        self._logger.info(f"Model loaded from {model_name} <<<")
        return model
    
    @abstractmethod
    def load_tokenizer(self, tokenizer_name):
        self._logger.info(f">>> Loading tokenizer from {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self._logger.info(f"Tokenizer loaded from {tokenizer_name} <<<")
        return tokenizer
 
    @abstractmethod
    def build_pipeline(self):
        self._logger.info(">>> Building pipeline")
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        self._logger.info("Pipeline built <<<")
        return pipe
    
    @abstractmethod
    def format_output(self, input, output):
        self._logger.info(">>> Formatting output")
        # remove the input from the output
        output = output.replace(input, '')
        # remove the punctuation that is unnecessary
        output = output.replace(':', ',').replace(';', ',')
        output = re.sub(r'[{}()\[\]]', '', output)
        self._logger.info("Output formatted <<<")
        return output.strip()
    
    @abstractmethod
    def format_prompt(self, prompt):
        return self.prompt_format.format(prompt)


class Chatbot(AbstractChatbot):
    def __init__(self, model_name, tokenizer_name, device='cuda:0', format='llama'):
        super().__init__(model_name, tokenizer_name, device, format)

    def _get_format(self, format):
        return super()._get_format(format)

    def _init_logger(self):
        return super()._init_logger()
    
    def _create_bnb_config(self):
        return super()._create_bnb_config()

    def load_model(self, model_name):
        return super().load_model(model_name)

    def load_tokenizer(self, tokenizer_name):
        return super().load_tokenizer(tokenizer_name)

    def build_pipeline(self):
        return super().build_pipeline()

    def format_prompt(self, prompt):
        return super().format_prompt(prompt)

    def format_output(self, input, output):
        return super().format_output(input, output)

    def generate(self, prompt):
        # format the prompt
        prompt_f = self.format_prompt(prompt)
        
        result = self.pipe(
            prompt_f,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=10,
            max_new_tokens=300,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

        output = self.format_output(prompt_f, result[0]['generated_text'])
        
        # output is valid, return it
        if output != '':
            return output
    
        # continue inferencing until a valid response is generated
        return self.generate(prompt)

class MergedChatbot(AbstractChatbot):
    def __init__(self, new_model, model_name, tokenizer_name, device='cuda:0', format='llama'):
        super().__init__(model_name, tokenizer_name, device, format)
        self.model = self.load_merged_model(new_model)

    def _get_format(self, format):
        return super()._get_format(format)

    def _init_logger(self):
        return super()._init_logger()
    
    def _create_bnb_config(self):
        return super()._create_bnb_config()

    def load_model(self, model_name):
        return super().load_model(model_name)

    def load_tokenizer(self, tokenizer_name):
        return super().load_tokenizer(tokenizer_name)

    def build_pipeline(self):
        return super().build_pipeline()

    def format_prompt(self, prompt):
        return super().format_prompt(prompt)

    def format_output(self, input, output):
        return super().format_output(input, output)

    def load_merged_model(self, new_model):
        self._logger.info(f">>> Merging Models")
        model = PeftModel.from_pretrained(self.model, new_model)
        model = model.merge_and_unload()
        self._logger.info(f"Models merged <<<")
        return model

    def generate(self, prompt):
        # format the prompt
        prompt_f = self.format_prompt(prompt)
        
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
        
        # output is valid, return it
        if output != '':
            return output
        # continue inferencing until a valid response is generated
        return self.generate(prompt)