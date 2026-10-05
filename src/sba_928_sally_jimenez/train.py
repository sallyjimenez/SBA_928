import csv

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments


model_name = "google/flan-t5-small"
training_file = "data/processed/training_data.csv"
validation_file = "data/processed/validation_data.csv"
model_folder = "models/flan-t5-ticket-classifier"
log_file = "outputs/training_log.csv"
configuration_file = "outputs/training_configuration.csv"

maximum_input_length = 256
maximum_target_length = 64


def read_examples(file_name):
    examples = []

    with open(file_name, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            examples.append(row)

    return examples


training_examples = read_examples(training_file)
validation_examples = read_examples(validation_file)

print("Training examples:", len(training_examples))
print("Validation examples:", len(validation_examples))


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_examples(examples):
    tokenized_examples = []

    for example in examples:
        input_text = "Instruction:\n" + example["instruction"]
        input_text += "\n\nContext:\n" + example["context"]

        tokenized_input = tokenizer(
            input_text,
            max_length=maximum_input_length,
            truncation=True,
        )

        tokenized_target = tokenizer(
            text_target=example["target"],
            max_length=maximum_target_length,
            truncation=True,
        )

        tokenized_example = {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "labels": tokenized_target["input_ids"],
        }

        tokenized_examples.append(tokenized_example)

    return tokenized_examples


training_data = tokenize_examples(training_examples)
validation_data = tokenize_examples(validation_examples)

print("Tokenized training examples:", len(training_data))
print("Tokenized validation examples:", len(validation_data))


# Load and configure the model
print("\nLoading the pretrained model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

training_settings = Seq2SeqTrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=0.00005,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    use_cpu=True,
    dataloader_pin_memory=False,
    optim="adamw_torch",
    seed=42,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_settings,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
    processing_class=tokenizer,
)


# Train and evaluate the model
print("\nStarting fine-tuning...")
training_result = trainer.train()
validation_result = trainer.evaluate()
print("Fine-tuning completed.")


# Save the model
trainer.save_model(model_folder)
tokenizer.save_pretrained(model_folder)
trainer.save_state()


# Save the training settings
configuration_rows = [
    ["setting", "value"],
    ["model", model_name],
    ["training examples", len(training_examples)],
    ["validation examples", len(validation_examples)],
    ["epochs", 2],
    ["training batch size", 2],
    ["validation batch size", 2],
    ["learning rate", 0.00005],
    ["maximum input length", maximum_input_length],
    ["maximum target length", maximum_target_length],
    ["random seed", 42],
    ["device", "CPU"],
]

with open(configuration_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(configuration_rows)


# Save the training and validation losses
log_columns = [
    "step",
    "epoch",
    "training_loss",
    "validation_loss",
    "learning_rate",
]

with open(log_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=log_columns)
    writer.writeheader()

    for log in trainer.state.log_history:
        writer.writerow(
            {
                "step": log.get("step", ""),
                "epoch": log.get("epoch", ""),
                "training_loss": log.get("loss", ""),
                "validation_loss": log.get("eval_loss", ""),
                "learning_rate": log.get("learning_rate", ""),
            }
        )


print("\nFinal training loss:", training_result.metrics.get("train_loss"))
print("Final validation loss:", validation_result.get("eval_loss"))
print("Saved model:", model_folder)
print("Saved training log:", log_file)
print("Saved configuration:", configuration_file)
