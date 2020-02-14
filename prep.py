import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm

class dataset():
    def __init__(self, base_dir = '/content/parrot_proj1/', size = (150, 150)):
        self.base_dir = base_dir
        self.size = size
    def train(self):

        images = []
        labels = []
        for folder in tqdm(os.listdir(self.base_dir+"train")):
            for file in os.listdir(self.base_dir+"train/"+folder):
                img  = cv2.imread(self.base_dir+"train/"+folder+"/"+file)
                img = cv2.resize(img, self.size)
                images.append(img)
                labels.append(folder)
        images = np.array(images, dtype="float32")
        labels = np.array(labels, dtype="int32")
        images - np.true_divide(images, 255)
        print(images.shape)
        return images, labels
    @classmethod
    def test(self):
        images = []
        index = []
        for file in tqdm(os.listdir(self.base_dir+"test")):
            img = cv2.imread(self.base_dir+"train/"+file)
            img = cv2.resize(img, self.size)
            images.append(img)
            index.append(os.path.splitext(file)[0])
        images = np.array(images, dtype="float32")
        images - np.true_divide(images, 255)
        return images, index
