from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from peft_utils import load_tokenizer, prepare_dataset, compute_metrics_fn


def setup_peft_model(base_model, tokenizer, num_labels):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def train_model(model, tokenizer, train_dataset, output_dir="."):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        evaluation_strategy="no",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn()
    )

    trainer.train()
    model.save_pretrained(output_dir)


def main():
    model_name = "gpt2"
    output_dir = "gpt2-agnews-lora"
    num_labels = 4

    tokenizer = load_tokenizer(model_name)
    train_dataset = prepare_dataset(tokenizer, split="train", subset_size=2000, data_name="ag_news")
    model = setup_peft_model(model_name, tokenizer, num_labels)

    train_model(model, tokenizer, train_dataset, output_dir)


if __name__ == "__main__":
    main()



