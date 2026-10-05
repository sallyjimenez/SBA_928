import csv

import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


evaluation_file = "data/processed/evaluation_data.csv"
output_file = "outputs/model_comparison.csv"
base_model_name = "google/flan-t5-small"
fine_tuned_model_folder = "models/flan-t5-ticket-classifier"

field_names = ["Ticket type", "Ticket subject", "Priority"]


def get_field(text, field_name):
    lowercase_text = text.lower()
    label = field_name.lower() + ":"
    start = lowercase_text.find(label)

    if start == -1:
        return ""

    start += len(label)
    end = len(text)

    for next_field in field_names:
        next_label = next_field.lower() + ":"
        next_position = lowercase_text.find(next_label, start)

        if next_position != -1 and next_position < end:
            end = next_position

    return text[start:end].strip()


def generate_responses(model_location, tokenizer, examples):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_location)
    model.eval()
    responses = []

    for example in examples:
        prompt = "Instruction:\n" + example["instruction"]
        prompt += "\n\nContext:\n" + example["context"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                do_sample=False,
            )

        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


def score_response(response, target):
    correct_fields = 0
    found_fields = 0

    for field_name in field_names:
        response_value = get_field(response, field_name)
        target_value = get_field(target, field_name)

        if response_value != "":
            found_fields += 1

        if response_value.lower() == target_value.lower():
            correct_fields += 1

    exact_match = 0
    clean_response = " ".join(response.lower().split())
    clean_target = " ".join(target.lower().split())

    if clean_response == clean_target:
        exact_match = 1

    correct_format = 0

    if found_fields == 3:
        correct_format = 1

    return correct_fields, exact_match, correct_format


# Read the evaluation examples
examples = []

with open(evaluation_file, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        examples.append(row)

print("Held-out evaluation examples:", len(examples))


# Run the same examples through both models
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("\nRunning the original model...")
base_responses = generate_responses(base_model_name, tokenizer, examples)

print("Running the fine-tuned model...")
fine_responses = generate_responses(
    fine_tuned_model_folder,
    tokenizer,
    examples,
)


# Compare the model responses
base_correct_fields = 0
fine_correct_fields = 0
base_exact_matches = 0
fine_exact_matches = 0
base_correct_formats = 0
fine_correct_formats = 0

column_names = [
    "example_number",
    "expected_target",
    "original_model_response",
    "fine_tuned_model_response",
    "original_correct_fields",
    "fine_tuned_correct_fields",
]

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=column_names)
    writer.writeheader()

    for number in range(len(examples)):
        target = examples[number]["target"]
        base_response = base_responses[number]
        fine_response = fine_responses[number]

        base_score = score_response(base_response, target)
        fine_score = score_response(fine_response, target)

        base_correct_fields += base_score[0]
        base_exact_matches += base_score[1]
        base_correct_formats += base_score[2]

        fine_correct_fields += fine_score[0]
        fine_exact_matches += fine_score[1]
        fine_correct_formats += fine_score[2]

        writer.writerow(
            {
                "example_number": number + 1,
                "expected_target": target,
                "original_model_response": base_response,
                "fine_tuned_model_response": fine_response,
                "original_correct_fields": base_score[0],
                "fine_tuned_correct_fields": fine_score[0],
            }
        )


# Display the totals
number_of_examples = len(examples)
number_of_fields = number_of_examples * 3

base_accuracy = base_correct_fields / number_of_fields * 100
fine_accuracy = fine_correct_fields / number_of_fields * 100

print("\nComparison results:")
print("\nOriginal model:")
print("Correct fields:", base_correct_fields, "out of", number_of_fields)
print("Field accuracy:", format(base_accuracy, ".2f") + "%")
print("Exact matches:", base_exact_matches, "out of", number_of_examples)
print("Correct output format:", base_correct_formats,
      "out of", number_of_examples)

print("\nFine-tuned model:")
print("Correct fields:", fine_correct_fields, "out of", number_of_fields)
print("Field accuracy:", format(fine_accuracy, ".2f") + "%")
print("Exact matches:", fine_exact_matches, "out of", number_of_examples)
print("Correct output format:", fine_correct_formats,
      "out of", number_of_examples)

print("\nSaved results to:")
print(output_file)
