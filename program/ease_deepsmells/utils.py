import torch
import numpy as np
import platform

from sklearn.model_selection import train_test_split

import sys

sys.path.insert(
    0, r"/Users/nguyenbinhminh/MasterUET/Thesis/DeepSmells-jsdata/program/dl_models"
)
import inputs
import input_data


def write_file(file, str):
    file = open(file, mode="a+")
    file.write(str)
    file.close()


def device():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    else:
        mps_device = torch.device("mps")

    return mps_device
    # if torch.cuda.is_available():
    #     print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}\n")
    #     device = torch.device("cuda")
    # else:
    #     print(f"\n[INFO] GPU not found. Using CPU: {platform.processor()}\n")
    #     device = torch.device("cpu")
    # return device


def concatenate(train_data, eval_data):
    return np.concatenate((train_data, eval_data), axis=0)


# Function get_all_data
def get_all_data(data_path, smell, train_validate_ratio=0.7):
    print("reading data...")

    if smell in ["ComplexConditional", "ComplexMethod"]:
        max_eval_samples = 150000  # for impl smells (methods)
    else:
        max_eval_samples = 50000  # for design smells (classes)

    # Load training and eval data
    train_data, train_labels, eval_data, eval_labels, max_input_length = (
        inputs.get_data(
            data_path,
            train_validate_ratio=train_validate_ratio,
            max_training_samples=5000,
            max_eval_samples=max_eval_samples,
        )
    )
    # train_data = train_data.reshape((len(train_labels), max_input_length))
    # eval_data = eval_data.reshape((len(eval_labels), max_input_length))
    print("train_data: " + str(len(train_data)))
    print("train_labels: " + str(len(train_labels)))
    print("eval_data: " + str(len(eval_data)))
    print("eval_labels: " + str(len(eval_labels)))
    print("reading data... done.")

    # Concat traindata and validdata
    data = concatenate(train_data, eval_data)
    labels = concatenate(train_labels, eval_labels)

    # Train-Test Split (Straitified Sampling)
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        data, labels, test_size=0.3, random_state=0, stratify=labels
    )

    return input_data.Input_data(
        train_data, train_labels, valid_data, valid_labels, max_input_length
    )


# def get_all_data(data_path, smell, train_validate_ratio=0.7):
#     print("reading data...")

#     if smell in ["ComplexConditional", "ComplexMethod"]:
#         max_eval_samples = 150000 # for impl smells (methods)
#     else:
#         max_eval_samples = 50000 # for design smells (classes)


#     # Load training and eval data
#     train_data, train_labels, eval_data, eval_labels, max_input_length = \
#         inputs.get_data(data_path,
#                         train_validate_ratio=train_validate_ratio,
#                         max_training_samples=5000,
#                         max_eval_samples=max_eval_samples)
#     # train_data = train_data.reshape((len(train_labels), max_input_length))
#     # eval_data = eval_data.reshape((len(eval_labels), max_input_length))
#     print("train_data: " + str(len(train_data)))
#     print("train_labels: " + str(len(train_labels)))
#     print("eval_data: " + str(len(eval_data)))
#     print("eval_labels: " + str(len(eval_labels)))
#     print("reading data... done.")

#     return input_data.Input_data(train_data, train_labels, eval_data, eval_labels, max_input_length)
