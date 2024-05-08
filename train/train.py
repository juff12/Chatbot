import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NousResearch/Meta-Llama-3-8B", help="name of the model to use")
    parser.add_argument("--dataset_name", type=str, default="data/datasets/test/test.json", help="name of the dataset to use")
    parser.add_argument("--new_model", type=str, default="outputs/chatbot/streamers/test/llama-3-8b-miniTest", help="output directory")
    
    # QLoRA parameters
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")

    # bitsandbytes parameters
    parser.add_argument("--use_4bit", type=bool, default=True, help="Activate 4-bit precision base model loading")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", help="Compute dtype for 4-bit base models")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    parser.add_argument("--use_nested_quant", type=bool, default=False, help="Activate nested quantization for 4-bit base models (double quantization)")

    # training Parameters
    parser.add_argument("--output_dir", type=str, default="output/chatbot/streamers/test/results", help="Output directory where the model predictions and checkpoints will be stored")
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--fp16", type=bool, default=False, help="Enable fp16 training (set bf16 to True with an A100)")
    parser.add_argument("--bf16", type=bool, default=False, help="Enable bf16 training (set bf16 to True with an A100)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate the gradients for")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm (gradient clipping)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate (AdamW optimizer)")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to apply to all layers except bias/LayerNorm weights")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate schedule")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of training steps (overrides num_train_epochs)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of steps for a linear warmup (from 0 to learning rate)")
    parser.add_argument("--group_by_length", type=bool, default=True, help="Group sequences into batches with same length")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X update steps")
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X update steps")

    # SFT parameters
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length to use")
    parser.add_argument("--packing", type=bool, default=False, help="Pack multiple short examples in the same input sequence to increase efficiency")

    return parser.parse_args()

def main():
    # get hyperparameters
    opt = args()

    # load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, opt.bnb_4bit_compute_dtype)

    # bitsandbytes configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=opt.use_4bit,
        bnb_4bit_quant_type=opt.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=opt.use_nested_quant,
    )

    # check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and opt.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # load the base model
    model = AutoModelForCausalLM.from_pretrained(
        opt.model_name,
        quantization_config=bnb_config,
        device_map='auto'
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load the config for QLoRA
    peft_config = LoraConfig(
        lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
        r=opt.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )


    # function for tokenizing the prompt
    generate_and_tokenize_prompt = lambda prompt: tokenizer(prompt['text'])

    # load the dataset
    data = load_dataset("json", data_files=opt.dataset_name)
    dataset = data['train'].map(generate_and_tokenize_prompt)

    # set the training parameters
    training_arguments = TrainingArguments(
        output_dir=opt.output_dir,
        num_train_epochs=opt.num_train_epochs,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        optim=opt.optim,
        save_steps=opt.save_steps,
        logging_steps=opt.logging_steps,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        fp16=opt.fp16,
        bf16=opt.bf16,
        max_grad_norm=opt.max_grad_norm,
        max_steps=opt.max_steps,
        warmup_ratio=opt.warmup_ratio,
        group_by_length=opt.group_by_length,
        lr_scheduler_type=opt.lr_scheduler_type,
        report_to="tensorboard"
    )

    # set the supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=opt.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=opt.packing,
    )

    # train the model
    trainer.train()

    # save the model
    trainer.model.save_pretrained(opt.new_model)

if __name__=="__main__":
    main()