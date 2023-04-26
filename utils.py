import json
import os
import torch
from config import DATA_DIR
import Levenshtein as lev

def load_all_words():
    with open(os.path.join(DATA_DIR, "all_words.json"), "r", encoding="utf-16") as f:
        all_words = json.load(f)
    return all_words

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    text_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        text_preds.append(remove_duplicates(tp))
    return text_preds

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def correct_text(text, all_words, max_distance=4):
    """Returns the corrected text based on edit distance threshold"""
    words = text.split()
    corrected_words = []
    for word in words:
        if word in all_words:
            corrected_words.append(word)
        else:
            candidates = []
            for candidate in all_words:
                if abs(len(word) - len(candidate)) > max_distance:
                    continue
                distance = lev.distance(word, candidate)
                if distance <= max_distance:
                    candidates.append((candidate, distance))
            if candidates:
                corrected_word = min(candidates, key=lambda x: x[1])[0]
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text