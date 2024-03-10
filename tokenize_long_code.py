import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", config=config)

# Define maximum sequence length and segment size
max_sequence_length = 512  # Maximum sequence length supported by BERT
segment_size = 128  # Segment size for splitting the code

# Example long JavaScript code
long_code = """
function mergeSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }

    const middle = Math.floor(arr.length / 2);
    const left = arr.slice(0, middle);
    const right = arr.slice(middle);

    return merge(
        mergeSort(left),
        mergeSort(right)
    );
}

function merge(left, right) {
    let resultArray = [], leftIndex = 0, rightIndex = 0;

    while (leftIndex < left.length && rightIndex < right.length) {
        if (left[leftIndex] < right[rightIndex]) {
            resultArray.push(left[leftIndex]);
            leftIndex++;
        } else {
            resultArray.push(right[rightIndex]);
            rightIndex++;
        }
    }

    return resultArray
        .concat(left.slice(leftIndex))
        .concat(right.slice(rightIndex));
}

const unsortedArray = [6, 5, 3, 1, 8, 7, 2, 4];
console.log('Original array:', unsortedArray);
const sortedArray = mergeSort(unsortedArray);
console.log('Sorted array:', sortedArray);
"""

# Tokenize the long code
tokenized_long_code = tokenizer.encode(long_code, add_special_tokens=False)

# Split the tokenized code into segments
segments = [
    tokenized_long_code[i : i + segment_size]
    for i in range(0, len(tokenized_long_code), segment_size)
]

# Initialize list to store representations of segments
segment_representations = []

# Iterate over segments and obtain representations
for segment in segments:
    # Ensure the segment fits within the maximum sequence length
    if len(segment) <= max_sequence_length:
        # Convert segment to tensor and add padding if necessary
        segment_tensor = torch.tensor(segment).unsqueeze(0)  # Add batch dimension
        if len(segment) < max_sequence_length:
            # Pad segment tensor to max_sequence_length
            segment_tensor = torch.nn.functional.pad(
                segment_tensor, (0, max_sequence_length - len(segment))
            )

        # Create attention mask
        attention_mask = torch.ones(segment_tensor.shape, dtype=torch.long)
        attention_mask[
            segment_tensor == tokenizer.pad_token_id
        ] = 0  # Set attention mask to 0 for padded tokens

        # Pass segment through BERT model to obtain representation
        with torch.no_grad():
            outputs = model(segment_tensor, attention_mask=attention_mask)
            segment_representation = outputs.last_hidden_state.mean(dim=1).squeeze(
                0
            )  # Average pooling over tokens
            segment_representations.append(segment_representation)

# Aggregate representations by averaging
global_representation = torch.stack(segment_representations).mean(dim=0)

# Define the number of output classes for the classifier
num_classes = 2  # Modify this as needed

# Ensure that global_representation is a 1D tensor
global_representation = global_representation.view(-1)

# Define a classifier on top of BERT
classifier = nn.Linear(global_representation.size(0), num_classes)

# Apply global representation to the classifier
output = classifier(global_representation)

print("Classifier output:", output)
