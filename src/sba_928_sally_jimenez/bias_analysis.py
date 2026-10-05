import csv
from collections import Counter


# Location of the customer-support dataset
raw_data_file = "data/raw/customer_support_tickets.csv"


# Read the dataset
tickets = []

with open(raw_data_file, newline="", encoding="utf-8-sig") as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        tickets.append(row)


# Create counters for gender and age groups
gender_counts = Counter()
age_group_counts = Counter()

# Count missing satisfaction ratings
missing_satisfaction_ratings = 0


for ticket in tickets:
    customer_gender = ticket["Customer Gender"]
    customer_age = int(ticket["Customer Age"])
    satisfaction_rating = ticket["Customer Satisfaction Rating"]

    gender_counts[customer_gender] += 1

    if customer_age < 25:
        age_group_counts["Under 25"] += 1
    elif customer_age < 45:
        age_group_counts["25-44"] += 1
    elif customer_age < 65:
        age_group_counts["45-64"] += 1
    else:
        age_group_counts["65 and older"] += 1

    if satisfaction_rating == "":
        missing_satisfaction_ratings += 1


# Display the results
print("Total tickets:", len(tickets))

print("\nCustomer gender counts:")

for gender, count in gender_counts.items():
    print(gender + ":", count)

print("\nCustomer age-group counts:")

for age_group, count in age_group_counts.items():
    print(age_group + ":", count)

print("\nMissing customer satisfaction ratings:")
print(missing_satisfaction_ratings, "out of", len(tickets))
