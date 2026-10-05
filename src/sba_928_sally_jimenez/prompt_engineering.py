import csv
from collections import Counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


raw_file = "data/raw/customer_support_tickets.csv"
prompt_file = "data/processed/prompt_variations.csv"
output_file = "outputs/prompt_engineering_outputs.csv"
model_name = "google/flan-t5-small"


# Read the dataset and prompts
tickets = []
prompts = []

with open(raw_file, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        tickets.append(row)

with open(prompt_file, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        prompts.append(row)

print("Customer-support tickets:", len(tickets))
print("Prompt variations:", len(prompts))


# Create the consumer-behavior context
type_counts = Counter()
subject_counts = Counter()

for ticket in tickets:
    type_counts[ticket["Ticket Type"]] += 1
    subject_counts[ticket["Ticket Subject"]] += 1

consumer_context = "Total tickets: " + str(len(tickets))
consumer_context += "\n\nTicket type counts:\n"

for ticket_type, count in type_counts.most_common():
    consumer_context += ticket_type + ": " + str(count) + "\n"

consumer_context += "\nFive most common ticket subjects:\n"

for subject, count in subject_counts.most_common(5):
    consumer_context += subject + ": " + str(count) + "\n"


# Create the market-trends context
monthly_totals = Counter()
monthly_types = {}
monthly_subjects = {}

for ticket in tickets:
    month = ticket["Date of Purchase"][:7]

    if month not in monthly_types:
        monthly_types[month] = Counter()
        monthly_subjects[month] = Counter()

    monthly_totals[month] += 1
    monthly_types[month][ticket["Ticket Type"]] += 1
    monthly_subjects[month][ticket["Ticket Subject"]] += 1

market_context = ""

for month in sorted(monthly_totals):
    top_type = monthly_types[month].most_common(1)[0]
    top_subject = monthly_subjects[month].most_common(1)[0]

    market_context += month + ": total tickets=" + str(monthly_totals[month])
    market_context += ", top ticket type=" + top_type[0]
    market_context += " (" + str(top_type[1]) + ")"
    market_context += ", top ticket subject=" + top_subject[0]
    market_context += " (" + str(top_subject[1]) + ")\n"


# Create the competitor-analysis context
products = ["Google Pixel", "iPhone", "Samsung Galaxy", "Sony Xperia"]
competitor_context = ""

for product in products:
    product_tickets = []

    for ticket in tickets:
        if ticket["Product Purchased"] == product:
            product_tickets.append(ticket)

    product_types = Counter()
    product_subjects = Counter()
    product_priorities = Counter()
    ratings = []

    for ticket in product_tickets:
        product_types[ticket["Ticket Type"]] += 1
        product_subjects[ticket["Ticket Subject"]] += 1
        product_priorities[ticket["Ticket Priority"]] += 1

        if ticket["Customer Satisfaction Rating"] != "":
            ratings.append(float(ticket["Customer Satisfaction Rating"]))

    top_type = product_types.most_common(1)[0]
    top_subject = product_subjects.most_common(1)[0]
    top_priority = product_priorities.most_common(1)[0]
    average_rating = sum(ratings) / len(ratings)

    competitor_context += product + ": total tickets=" + str(len(product_tickets))
    competitor_context += ", top ticket type=" + top_type[0]
    competitor_context += " (" + str(top_type[1]) + ")"
    competitor_context += ", top ticket subject=" + top_subject[0]
    competitor_context += " (" + str(top_subject[1]) + ")"
    competitor_context += ", top priority=" + top_priority[0]
    competitor_context += " (" + str(top_priority[1]) + ")"
    competitor_context += ", average satisfaction="
    competitor_context += format(average_rating, ".2f")
    competitor_context += "/5 from " + str(len(ratings)) + " rated tickets\n"


# Match each research area with its context
contexts = {
    "Consumer behavior": consumer_context,
    "Market trends": market_context,
    "Competitor analysis": competitor_context,
}


# Load the original model
print("\nLoading the original model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()


# Run the prompts and save their responses
columns = [
    "variation_id",
    "source_prompt_id",
    "research_area",
    "prompt_style",
    "prompt",
    "data_context",
    "base_model_response",
]

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for prompt_row in prompts:
        research_area = prompt_row["research_area"]
        data_context = contexts[research_area]
        complete_prompt = prompt_row["prompt"]
        complete_prompt += "\n\nData context:\n" + data_context

        model_input = tokenizer(
            complete_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.no_grad():
            output = model.generate(
                input_ids=model_input["input_ids"],
                attention_mask=model_input["attention_mask"],
                max_new_tokens=160,
                do_sample=False,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        result = {
            "variation_id": prompt_row["variation_id"],
            "source_prompt_id": prompt_row["source_prompt_id"],
            "research_area": research_area,
            "prompt_style": prompt_row["prompt_style"],
            "prompt": prompt_row["prompt"],
            "data_context": data_context,
            "base_model_response": response,
        }

        writer.writerow(result)
        print("Completed:", prompt_row["variation_id"])

print("\nSaved responses to:")
print(output_file)
