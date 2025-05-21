import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def load_and_prepare_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "gemma-2b",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    return model

def load_train_eval_datasets(train_path="data/train.json", val_path="data/val.json"):
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path
    })
    return dataset["train"], dataset["validation"]

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def run_training():
    model = load_and_prepare_model()
    train_dataset, eval_dataset = load_train_eval_datasets()

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=100,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        fp16=True
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
