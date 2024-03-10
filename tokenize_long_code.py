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


# output tensor([[ 0.2979, -0.4207],
#         [ 0.3150, -0.4615],
#         [ 0.3035, -0.4433],
#         [ 0.2827, -0.4337],
#         [ 0.2942, -0.4183],
#         [ 0.3082, -0.4842],
#         [ 0.3048, -0.4658],
#         [ 0.3129, -0.4345],
#         [ 0.2859, -0.4459],
#         [ 0.3019, -0.4190],
#         [ 0.3175, -0.5499],
#         [ 0.2976, -0.4466],
#         [ 0.3041, -0.4733],
#         [ 0.3207, -0.4852],
#         [ 0.2964, -0.4529],
#         [ 0.3108, -0.4313],
#         [ 0.3067, -0.4011],
#         [ 0.3053, -0.4849],
#         [ 0.3323, -0.4951],
#         [ 0.3010, -0.4522],
#         [ 0.2931, -0.4339],
#         [ 0.2871, -0.4425],
#         [ 0.3158, -0.4587],
#         [ 0.2970, -0.4388],
#         [ 0.2654, -0.4802],
#         [ 0.3050, -0.4664],
#         [ 0.2994, -0.4256],
#         [ 0.3188, -0.4620],
#         [ 0.2808, -0.4683],
#         [ 0.2903, -0.4584],
#         [ 0.2699, -0.4161],
#         [ 0.2963, -0.4278],
#         [ 0.3101, -0.4609],
#         [ 0.2975, -0.4381],
#         [ 0.3311, -0.4743],
#         [ 0.3122, -0.4466],
#         [ 0.3320, -0.4796],
#         [ 0.3365, -0.4658],
#         [ 0.3082, -0.4140],
#         [ 0.2902, -0.4003],
#         [ 0.3096, -0.4533],
#         [ 0.2976, -0.4498],
#         [ 0.3039, -0.4122],
#         [ 0.2986, -0.4964],
#         [ 0.2694, -0.4261],
#         [ 0.2809, -0.4640],
#         [ 0.3161, -0.5161],
#         [ 0.2893, -0.4323],
#         [ 0.2961, -0.4244],
#         [ 0.2778, -0.4893],
#         [ 0.3109, -0.4268],
#         [ 0.2866, -0.4272],
#         [ 0.3078, -0.4119],
#         [ 0.3387, -0.5253],
#         [ 0.3249, -0.4828],
#         [ 0.2863, -0.4928],
#         [ 0.2870, -0.4156],
#         [ 0.3193, -0.4848],
#         [ 0.3298, -0.4601],
#         [ 0.3148, -0.4145],
#         [ 0.2957, -0.4630],
#         [ 0.2791, -0.4636],
#         [ 0.3021, -0.4597],
#         [ 0.3059, -0.4758],
#         [ 0.3421, -0.4718],
#         [ 0.2999, -0.4390],
#         [ 0.2473, -0.4630],
#         [ 0.3100, -0.4615],
#         [ 0.3014, -0.4220],
#         [ 0.3117, -0.4341],
#         [ 0.2679, -0.4285],
#         [ 0.2472, -0.4127],
#         [ 0.3047, -0.4291],
#         [ 0.2784, -0.4537],
#         [ 0.2846, -0.4700],
#         [ 0.3052, -0.4150],
#         [ 0.3158, -0.4708],
#         [ 0.3222, -0.4563],
#         [ 0.2832, -0.4338],
#         [ 0.3365, -0.4562],
#         [ 0.2892, -0.4076],
#         [ 0.3033, -0.4262],
#         [ 0.2774, -0.4003],
#         [ 0.3027, -0.4411],
#         [ 0.2973, -0.4542],
#         [ 0.2818, -0.5159],
#         [ 0.3068, -0.4113],
#         [ 0.3243, -0.4527],
#         [ 0.3105, -0.4429],
#         [ 0.2774, -0.4547],
#         [ 0.2909, -0.4513],
#         [ 0.3229, -0.4670],
#         [ 0.3259, -0.4797],
#         [ 0.3196, -0.4638],
#         [ 0.3332, -0.5003],
#         [ 0.3235, -0.4288],
#         [ 0.2685, -0.4557],
#         [ 0.3364, -0.4342],
#         [ 0.2905, -0.4639],
#         [ 0.2756, -0.4335],
#         [ 0.2498, -0.4465],
#         [ 0.3309, -0.5196],
#         [ 0.3100, -0.5231],
#         [ 0.3095, -0.4472],
#         [ 0.3106, -0.4594],
#         [ 0.3387, -0.4447],
#         [ 0.2942, -0.3938],
#         [ 0.3165, -0.4648],
#         [ 0.3458, -0.4591],
#         [ 0.2609, -0.4097],
#         [ 0.3228, -0.4505],
#         [ 0.2928, -0.4839],
#         [ 0.2910, -0.4437],
#         [ 0.3010, -0.4411],
#         [ 0.3097, -0.5239],
#         [ 0.3130, -0.4958],
#         [ 0.2938, -0.4341],
#         [ 0.2783, -0.4464],
#         [ 0.2503, -0.4591],
#         [ 0.3015, -0.4709],
#         [ 0.2911, -0.4086],
#         [ 0.2833, -0.4675],
#         [ 0.3233, -0.4345],
#         [ 0.2934, -0.4782],
#         [ 0.2831, -0.4450],
#         [ 0.2980, -0.4220],
#         [ 0.2760, -0.4371],
#         [ 0.2952, -0.4094]], device='mps:0', grad_fn=<LinearBackward0>)
