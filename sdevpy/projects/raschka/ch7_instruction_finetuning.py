import json
from sdevpy.llms.datasets import *


######## Inspect dataset #################################################
file = r"C:\\temp\\llms\\datasets\\instruction-data.json"
with open(file, "r") as f:
    data = json.load(f)

print("Number of entries:", len(data))
print("Example entry\n", data[999])

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# Prepare training, test and validation sets
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))
