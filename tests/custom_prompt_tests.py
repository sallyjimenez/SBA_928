import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


model_folder = "models/flan-t5-ticket-classifier"

name = input("What is your name? ")
product = input("What product do you need help with? ")
issue = input("How can I help you? ")

instruction = (
    "Classify the customer-support ticket. "
    "Identify its ticket type, ticket subject, and priority."
)

model_input = "Instruction:\n" + instruction
model_input += "\n\nContext:\nProduct: " + product
model_input += "\nTicket description: " + issue

print("\nLoading the model...")

tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)
model.eval()

tokens = tokenizer(
    model_input,
    return_tensors="pt",
    truncation=True,
)

with torch.no_grad():
    output = model.generate(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        max_new_tokens=64,
        do_sample=False,
    )

response = tokenizer.decode(
    output[0],
    skip_special_tokens=True,
)

print("\nThank you, " + name + ".")
print("Ticket classification:")
print(response)
