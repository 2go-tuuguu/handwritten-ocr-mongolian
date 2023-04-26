import os
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_DIR

# dataset_path = '../datasets/training/Trainset.mat'
# data = 'train_data'
# save_dir = 'train'

dataset_path = 'Testset_I.mat'
data = 'test_data'
save_dir = 'test_1'

# dataset_path = 'Testset_II.mat'
# data = 'test_data3'
# save_dir = 'test_2'

# Create directory to save images if it doesn't already exist
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(DATA_DIR, save_dir))

# load dataset file
dataset = loadmat(os.path.join(DATA_DIR, dataset_path))
count = 0
# save all images as inverted PNG files in the 'train' directory with progress bar
for i in tqdm(range(len(dataset[data]))):
    img = dataset[data][i][0]
    # check if image has non zero values
    if np.count_nonzero(img) == 0:
        continue
    
    inverted_img = np.max(img) - img

    # Scale the image data to the range [0, 255]
    inverted_img = (inverted_img - np.min(inverted_img)) * (255.0 / (np.max(inverted_img) - np.min(inverted_img)))
    # Convert the image data to uint8 format
    inverted_img = inverted_img.astype(np.uint8)
    # Create an Image object from the image data
    image = Image.fromarray(inverted_img)
    # Save the image as a .png file
    image.save(f'{save_dir}/image_{count}.png')
    count += 1