import torch
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
import utils
import argparse

def argument():
    parser = argparse.ArgumentParser(description='Hyperparameter')

    parser.add_argument('--data', type=str, required=True, default="../../data/tokenizer_cs", help='path to dataset')
    parser.add_argument('--smell', type=str, required=True, help='smell to make experiments, we have 4 smells: "ComplexMethod", "ComplexConditional", "FeatureEnvy", "MultifacetedAbstraction"')

    parser.add_argument('--nb_epochs', type=int, required=True, default=60, help='The number of epochs')
    parser.add_argument('--train_batchsize', type=int, required=True, default=128, help='Train batch size')
    parser.add_argument('--valid_batchsize', type=int, required=True, default=128, help='Valid batch size')
    parser.add_argument('--lr', type=float, required=True, default=0.03, help='learning rate')
    parser.add_argument('--threshold', type=float, required=True, default=0.5, help='Threshold to classify')

    parser.add_argument('--model', type=str, required=True, default="../../data/tokenizer_cs", help='Model to train, we have "DeepSmells" and "DeepSmell-BiLSTM"')
    parser.add_argument('--hidden_size_lstm', type=int, required=True, help='Hidden size of lstm networks')

    parser.add_argument('--checkpoint_dir', type=str, required=True, default="./checkpoint/", help='path to checkpoint dir')
    parser.add_argument('--tracking_dir', type=str, required=True, default="./tracking/", help='path to tracking dir')

    print(f"{'='*30}{'='*30}")
    parser.print_help()
    print(f"{'='*30}{'='*30}")

    args = parser.parse_args()

    return args

class Config:
    # Data
    TOKENIZER_OUT_CS_PATH = "../../data/tokenizer_cs"
    SMELLS = ["ComplexMethod", "ComplexConditional", "FeatureEnvy", "MultifacetedAbstraction"]
    DIM = '1d'

    # setup DEVICE
    DEVICE = utils.device() 

    # Config train
    NB_EPOCHS = 60
    TRAIN_BS = 128
    VALID_BS = 128
    LR = 0.03
    SCALER = GradScaler()
    THRESHOLD = 0.5

    # Model
    MODEL = "LSTM"
    HIDDEN_LSTM = 1000

    # Config dir
    CHECKPOINT_DIR = "./checkpoint/"
    TRACKING_DIR = "./tracking/"