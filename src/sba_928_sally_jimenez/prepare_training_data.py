import csv
import random


raw_file = "data/raw/customer_support_tickets.csv"
training_file = "data/processed/training_data.csv"
validation_file = "data/processed/validation_data.csv"
evaluation_file = "data/processed/evaluation_data.csv"

random.seed(42)

instruction = (
    "Classify the customer-support ticket. "
    "Identify its ticket type, ticket subject, and priority."
)

ticket_types = [
    "Refund request",
    "Technical issue",
    "Cancellation request",
    "Product inquiry",
    "Billing inquiry",
]

examples_by_type = {}

for ticket_type in ticket_types:
    examples_by_type[ticket_type] = []

used_descriptions = []


# Read and clean the original tickets
with open(raw_file, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for ticket in reader:
        ticket_type = ticket["Ticket Type"]
        product = ticket["Product Purchased"]
        description = ticket["Ticket Description"]
        description = description.replace("{product_purchased}", product)

        safe_words = []

        for word in description.split():
            digit_count = 0

            for character in word:
                if character.isdigit():
                    digit_count += 1

            if "@" in word:
                safe_words.append("[EMAIL REMOVED]")
            elif digit_count >= 4:
                safe_words.append("[NUMBER REMOVED]")
            else:
                safe_words.append(word)

        description = " ".join(safe_words)[:500]

        if description == "" or description in used_descriptions:
            continue

        used_descriptions.append(description)

        context = "Product: " + product
        context += "\nTicket description: " + description

        target = "Ticket type: " + ticket_type
        target += "\nTicket subject: " + ticket["Ticket Subject"]
        target += "\nPriority: " + ticket["Ticket Priority"]

        example = {
            "instruction": instruction,
            "context": context,
            "target": target,
        }

        examples_by_type[ticket_type].append(example)


# Select examples from each ticket type
selected_examples = []

print("Available examples after cleaning:")

for ticket_type in ticket_types:
    examples = examples_by_type[ticket_type]
    print(ticket_type + ":", len(examples))

    random.shuffle(examples)

    for example in examples[:50]:
        selected_examples.append(example)

random.shuffle(selected_examples)


# Divide the examples into three datasets
training_examples = selected_examples[:200]
validation_examples = selected_examples[200:225]
evaluation_examples = selected_examples[225:250]


def save_examples(file_name, examples):
    columns = ["instruction", "context", "target"]

    with open(file_name, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(examples)


save_examples(training_file, training_examples)
save_examples(validation_file, validation_examples)
save_examples(evaluation_file, evaluation_examples)


print("\nSaved data splits:")
print("Training examples:", len(training_examples))
print("Validation examples:", len(validation_examples))
print("Held-out evaluation examples:", len(evaluation_examples))

print("\nExample structure:")
print("Instruction:")
print(training_examples[0]["instruction"])
print("\nContext:")
print(training_examples[0]["context"])
print("\nTarget:")
print(training_examples[0]["target"])
