import re

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def calculate_tpr_fpr(true_labels, predicted_labels):
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Extract values from confusion matrix
    TP = cm[1, 1]  # True Positive
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative

    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    tpr = TP / (TP + FN)  # TPR = TP / (TP + FN)
    fpr = FP / (FP + TN)  # FPR = FP / (FP + TN)

    return tpr, fpr


# Load the CSV file
file_path = "path\\to\\output\\csv"  # Path to csv file containing top-n matches

# Sample csv path
# file_path = "<path to dataset folder>\\LFWdataset\\output.csv"

data = pd.read_csv(file_path)

# Extract ground truth names (base names without numeric suffixes)
data["ground_truth"] = data["input_image"].apply(
    lambda x: re.match(r"(.+?)_\d+\.jpg", x).group(1)
)

# Extract ground truth names by splitting on the underscore and taking the first part
data["ground_truth"] = data["input_image"].apply(lambda x: x.split("_")[0])


# Define a function to check if ground truth matches any predicted name
def check_match(row):
    # Check for NaN in 'result' column
    if pd.isna(row["result"]):
        return False  # No match if 'result' is NaN
    # Split the result by spaces to get individual file names
    predicted_names = row["result"].split()
    # Extract base names from each filename
    predicted_base_names = [
        re.match(r"(.+?)_\d+\.jpg", name).group(1) for name in predicted_names
    ]

    # Extract base name from the input image
    ground_truth = re.match(r"(.+?)_\d+\.jpg", row["input_image"]).group(1)

    if not row["true_label"]:
        if len(predicted_base_names) > 0:
            return True
        else:
            return False
    # Check if the ground truth matches any of the filtered base names
    return ground_truth in predicted_base_names


# Determine the midpoint of the DataFrame
midpoint = len(data) // 2

# Set `true_label` column
data["true_label"] = [True] * midpoint + [False] * (len(data) - midpoint)

# Apply the function to each row to create a 'predicted' column with boolean values
false_positive_rates = []
true_positive_rates = []

data["predicted"] = data.apply(lambda row: check_match(row), axis=1)
tpr, fpr = calculate_tpr_fpr(data["true_label"], data["predicted"])

# Calculate metrics
accuracy = accuracy_score(data["true_label"], data["predicted"])
precision = precision_score(data["true_label"], data["predicted"])
recall = recall_score(data["true_label"], data["predicted"])
f1 = f1_score(data["true_label"], data["predicted"])


# Output the results
print("Accuracy:", accuracy, "Precision:", precision, "Recall:", recall)
