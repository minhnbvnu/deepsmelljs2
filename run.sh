python ./program/ease_deepsmells/main.py \
    --data "/content/drive/MyDrive/LabRISE/CodeSmell/DeepSmells/data/tokenizer_cs"\
    --smell "ComplexMethod" \
    --model "DeepSmells" \
    --nb_epochs  60 \
    --train_batchsize 128 \
    --valid_batchsize 128 \
    --lr 0.03 \
    --threshold 0.5 \
    --hidden_size_lstm 100 \
    --checkpoint_dir "/content/drive/MyDrive/LabRISE/CodeSmell/DeepSmells/program/ease_deepsmells/checkpoint" \
    --tracking_dir "/content/drive/MyDrive/LabRISE/CodeSmell/DeepSmells/program/ease_deepsmells/tracking/"