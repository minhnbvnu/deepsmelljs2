from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 10)
print(result)
"""
tokenizer.model_max_length = 15
code_tokens = [tokenizer.tokenize(line) for line in code.split("\n") if line.strip()]
print("code_tokens", code_tokens)
tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
print("tokens_ids", tokens_ids)
print(
    "model input",
    torch.tensor(tokens_ids),
    torch.tensor(tokens_ids)[None, :],
    torch.tensor(tokens_ids)[None, :].shape,
)
context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
print("context_embeddings", context_embeddings, context_embeddings.shape)

# code_tokens ['def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb']
# tokens_ids [9232, 19220, 1640, 102, 6, 428, 3256, 114, 10, 15698, 428, 35, 671, 10, 1493, 671, 741]
# model input tensor([ 9232, 19220,  1640,   102,     6,   428,  3256,   114,    10, 15698,
#           428,    35,   671,    10,  1493,   671,   741]) tensor([[ 9232, 19220,  1640,   102,     6,   428,  3256,   114,    10, 15698,
#            428,    35,   671,    10,  1493,   671,   741]]) torch.Size([1, 17])
# context_embeddings tensor([[[-0.2317,  0.1989, -0.0010,  ..., -0.1436, -0.4442,  0.3700],
#          [-0.7914,  0.2893, -0.3215,  ..., -0.4757, -0.7815,  0.5896],
#          [-0.7671,  0.2561, -0.1194,  ..., -0.3160, -0.2157, -0.1579],
#          ...,
#          [ 0.0183,  0.1544,  0.2515,  ...,  0.2120,  0.0073,  0.0983],
#          [-0.5575,  0.0548,  0.2625,  ..., -0.1846, -0.4276,  0.6065],
#          [-0.4035,  0.3386,  0.2415,  ..., -0.2956, -0.7370,  0.1431]]],
#        grad_fn=<NativeLayerNormBackward0>) torch.Size([1, 17, 768])
