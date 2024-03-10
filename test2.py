import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


# Generate synthetic JavaScript functions
def generate_javascript_function():
    # Generate a random length for the function
    length = random.randint(5, 20)
    # Generate random tokens for the function
    function_tokens = [random.randint(0, 10000) for _ in range(length)]
    return function_tokens


# Generate a larger dataset
num_samples = 100  # Number of samples in the dataset
tokenized_functions = [generate_javascript_function() for _ in range(num_samples)]
labels = [random.randint(0, 1) for _ in range(num_samples)]  # Generate random labels

# Split data into train and test sets
train_tokens, test_tokens, train_labels, test_labels = train_test_split(
    tokenized_functions, labels, test_size=0.2, random_state=42
)


# Define a custom dataset class
class JavaScriptDataset(Dataset):
    def __init__(self, tokenized_functions, labels, max_length=20):
        self.tokenized_functions = tokenized_functions
        self.labels = labels
        self.max_length = max_length

    def __getitem__(self, idx):
        tokenized_function = self.tokenized_functions[idx]
        # Pad or truncate the tokenized function to max_length
        tokenized_function = tokenized_function[: self.max_length] + [0] * (
            self.max_length - len(tokenized_function)
        )
        item = {
            "input_ids": torch.tensor(tokenized_function),
            "labels": torch.tensor(self.labels[idx]),
        }
        return item

    def __len__(self):
        return len(self.labels)


# Create datasets
train_dataset = JavaScriptDataset(train_tokens, train_labels)
test_dataset = JavaScriptDataset(test_tokens, test_labels)

# Define batch size and create DataLoader
batch_size = 4  # Adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define a custom classifier class
class CustomRobertaClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(CustomRobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classification_head = torch.nn.Linear(
            self.roberta.config.hidden_size, num_labels
        )

    def forward(self, input_ids):
        print("input_ids", input_ids, input_ids.shape, type(input_ids))
        outputs = self.roberta(input_ids)
        logits = outputs.last_hidden_state[
            :, 0, :
        ]  # Use CLS token representation for classification
        logits = self.classification_head(logits)
        return logits


# Initialize the model
num_labels = 2  # Assuming binary classification
model = CustomRobertaClassifier(num_labels)

# Fine-tune the model
device = torch.device("mps")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0.0
    train_pbar = tqdm(
        enumerate(train_loader), total=len(train_loader)
    )  # Create tqdm progress bar
    for step, batch in train_pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}"
        )  # Update progress bar description

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
