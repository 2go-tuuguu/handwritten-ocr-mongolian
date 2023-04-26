import os
import glob
import torch
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import decode_predictions, load_all_words, correct_text
from config import (
    DEVICE,
    DATA_DIR,
    TEST_DIR,
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    CHECKPOINT_PATH,
)
from dataset import MongolianOCRDataset
from model import HandwrittenMongolianModel
from engine import eval_fn
import Levenshtein as lev
from PIL import Image

# label encoder
targets_file = os.path.join(os.path.join(DATA_DIR, "Trainset_label.txt"))
with open(targets_file, "r", encoding='utf_16') as f:
    targets_orig = [line.strip() for line in f.readlines()]
targets = [[c for c in x] for x in targets_orig]
targets_flat = [c for clist in targets for c in clist]

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flat)

# images and targets
image_files = glob.glob(os.path.join(DATA_DIR, TEST_DIR, "*.png"))
image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
targets_file = os.path.join(os.path.join(DATA_DIR, TEST_DIR, "Testset_I_label.txt"))
with open(targets_file, "r", encoding='utf_16') as f:
    targets_orig = [line.strip() for line in f.readlines()]
targets = [[c for c in x] for x in targets_orig]

targets_enc = [lbl_enc.transform(x) for x in targets]
targets_enc_padded = pad_sequence([torch.tensor(x) + 1 for x in targets_enc], batch_first=True, padding_value=0)

test_dataset = MongolianOCRDataset(
    image_paths=image_files,
    targets=targets_enc_padded,
    resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,

)

all_words = load_all_words()
model = HandwrittenMongolianModel(num_chars=len(lbl_enc.classes_))
if os.path.exists(os.path.join(DATA_DIR, CHECKPOINT_PATH)):
    if DEVICE == "cpu":
        checkpoint = torch.load(os.path.join(DATA_DIR, CHECKPOINT_PATH), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(DATA_DIR, CHECKPOINT_PATH))
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    EPOCH_START = checkpoint['epoch']
    print(f'Checkpoint loaded at {EPOCH_START} epoch.')

valid_preds, test_loss = eval_fn(model, test_loader)
valid_text_preds = []

for vp in valid_preds:
    current_preds = decode_predictions(vp, lbl_enc)
    valid_text_preds.extend(current_preds)
corrected_preds = [correct_text(text, all_words) for text in valid_text_preds]
valid_text_preds = corrected_preds

# show first 5 predictions
for i in range(5):
    data = test_dataset[i]
    image = data['images'][0]
    #  show image from tensor
    image = Image.fromarray(image.numpy())
    image.show()
    print(f'Prediction: {valid_text_preds[i]}, target: {targets_orig[i]}')

# Calculate accuracy
accuracy = metrics.accuracy_score(targets_orig, valid_text_preds)
print("Accuracy: ", accuracy)

# Calculate CER and AED
cer = 0.0
aed = 0.0
for i in range(len(targets_orig)):
    cer += lev.distance(targets_orig[i], valid_text_preds[i]) / len(targets_orig[i])
    aed += lev.distance(targets_orig[i], valid_text_preds[i])
cer /= len(targets_orig)
aed /= len(targets_orig)
print("CER: ", cer)
print("AED: ", aed)