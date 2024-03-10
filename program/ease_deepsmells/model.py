import torch
import torch.nn as nn
import numpy as np
import math
from transformers import RobertaTokenizer, RobertaModel


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        kernel_size,
        input_size_lstm,
        hidden_size_lstm,
        input_dim=1,
        conv_dim1=16,
        conv_dim2=32,
        hidden_fc1=64,
        hidden_fc2=64,
        num_classes=1,
    ):
        super(CNN_LSTM, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim, out_channels=conv_dim1, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(conv_dim1, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
            nn.Conv1d(
                in_channels=conv_dim1, out_channels=conv_dim2, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(conv_dim2, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
        )

        self.lstm = nn.LSTM(
            input_size=input_size_lstm, hidden_size=hidden_size_lstm, batch_first=True
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size_lstm, hidden_fc1),
            nn.ReLU(),
            nn.Linear(hidden_fc1, hidden_fc2),
            nn.ReLU(),
            nn.Linear(hidden_fc2, num_classes),
        )

    def forward(self, text):
        out = self.conv_layers(text)
        out, (hidden, cell) = self.lstm(out)

        hidden = torch.squeeze(hidden)

        out = self.dense_layers(hidden)
        return out


class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        kernel_size,
        input_size_lstm,
        hidden_size_lstm,
        input_dim=1,
        conv_dim1=16,
        conv_dim2=32,
        hidden_fc1=64,
        hidden_fc2=64,
        num_classes=1,
    ):
        super(CNN_BiLSTM, self).__init__()
        self.hidden_size_lstm = hidden_size_lstm

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim, out_channels=conv_dim1, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(conv_dim1, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
            nn.Conv1d(
                in_channels=conv_dim1, out_channels=conv_dim2, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(conv_dim2, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
        )

        self.bilstm = nn.LSTM(
            input_size=input_size_lstm,
            hidden_size=hidden_size_lstm,
            batch_first=True,
            bidirectional=True,
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size_lstm * 2, hidden_fc1),
            nn.ReLU(),
            nn.Linear(hidden_fc1, hidden_fc2),
            nn.ReLU(),
            nn.Linear(hidden_fc2, num_classes),
        )

    def forward(self, text):
        out = self.conv_layers(text)
        out, (hidden, cell) = self.bilstm(out)

        hidden = torch.reshape(
            torch.transpose(hidden, 0, 1), (-1, self.hidden_size_lstm * 2)
        )

        out = self.dense_layers(hidden)
        return out


def size_output_conv(Lin, padding=0, dilation=1, kernel_size=3, stride=1):
    return math.floor(
        (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def calculate_size_lstm(input_size, kernel_size):
    # out conv 1
    out = size_output_conv(input_size, kernel_size=kernel_size)
    # out maxpool 1
    out = size_output_conv(out, kernel_size=kernel_size, stride=kernel_size)

    # out conv 2
    out = size_output_conv(out, kernel_size=kernel_size)
    # out maxpool 2
    out = size_output_conv(out, kernel_size=kernel_size, stride=kernel_size)

    return out


class CNN_LSTM_CodeBERT(nn.Module):
    def __init__(self, hidden_size):
        super(CNN_LSTM_CodeBERT, self).__init__()

        self.roberta = RobertaModel.from_pretrained(
            "microsoft/codebert-base",
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/codebert-base",
        )
        self.classifier = nn.Linear(hidden_size, 2)
        self.rnn = nn.RNN(
            self.roberta.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, texts):  # assuming texts is a tensor
        max_sequence_length = 512  # Maximum sequence length supported by RoBERTa
        segment_size = 128  # Segment size for splitting the code

        # Break the long_code_tensor into segments
        segments = [
            texts[:, i : i + segment_size]
            for i in range(0, texts.shape[1], segment_size)
        ]

        # Initialize list to store representations of segments
        segment_representations = []

        # Iterate over segments and obtain representations
        for segment in segments:
            # Pass the segment through RoBERTa model
            with torch.no_grad():
                outputs = self.roberta(input_ids=segment)
                segment_representation = outputs.last_hidden_state.mean(dim=1).squeeze(
                    0
                )  # Average pooling over tokens
                segment_representations.append(segment_representation)

        # Stack segment representations along the sequence dimension
        stacked_representations = torch.stack(segment_representations, dim=1)

        # Pass the stacked representations through the RNN
        rnn_output, _ = self.rnn(stacked_representations)

        # Apply global representation to the classifier
        output = self.classifier(
            rnn_output[:, -1, :]
        )  # Take the last hidden state of the RNN
        print("output", output)
        return output
