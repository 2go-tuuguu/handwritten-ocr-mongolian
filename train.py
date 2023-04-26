import os
import glob
import torch
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence

from config import (
    DEVICE,
    DATA_DIR,
    TRAIN_DIR,
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    EPOCH_END,
    EPOCH_START,
    CHECKPOINT_PATH,
)
from utils import decode_predictions, weights_init
import engine
from dataset import MongolianOCRDataset
from model import HandwrittenMongolianModel

def run_training():
    torch.set_printoptions(profile="full")
    image_files = glob.glob(os.path.join(DATA_DIR, TRAIN_DIR, "*.png"))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    targets_file = os.path.join(os.path.join(DATA_DIR, "Trainset_label.txt"))
    with open(targets_file, "r", encoding='utf_16') as f:
        targets_orig = [line.strip() for line in f.readlines()]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc_padded = pad_sequence([torch.tensor(x) + 1 for x in targets_enc], batch_first=True, padding_value=0)

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files, targets_enc_padded, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = MongolianOCRDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_dataset = MongolianOCRDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )
    
    model = HandwrittenMongolianModel(num_chars=len(lbl_enc.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        EPOCH_START = checkpoint['epoch']
        print(f'Checkpoint loaded at {EPOCH_START} epoch.')
    else:
        model.apply(weights_init)
        model.to(DEVICE)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        train_loss = train_fn(model, train_loader, optimizer)
        torch.cuda.empty_cache()
        valid_preds, test_loss = eval_fn(model, test_loader)
        # print(valid_preds[:10])
        valid_text_preds = []
        # print(valid_preds[0].shape)
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_text_preds.extend(current_preds)
        combined = list(zip(test_targets_orig, valid_text_preds))
        accuracy = metrics.accuracy_score(test_targets_orig, valid_text_preds)
        if epoch % 10 == 0:
            print(combined[:10])
            print(
                f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
            )
        # write to file for each epoch
        with open("results_train.txt", "a") as f:
            f.write(f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}\n")
            f.write(str(combined[:10]) + '\n')
        scheduler.step(test_loss)
        if epoch % 20 == 0:
            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler.state_dict()}
            torch.save(checkpoint, os.path.join(DATA_DIR, f'checkpoint_{epoch}.pth'))

if __name__ == "__main__":
    run_training()