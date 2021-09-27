import os
import random
import shutil
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataLoader(Dataset):
    def __init__(self, dir_path, transform):
        self.images_dict = {}
        self.id2image = {}
        self.imgName2label = {}
        self.labels = None
        self.dir_path = dir_path
        self.transform = transform
        self.load_images()

    def load_images(self):
        self.labels = os.listdir(self.dir_path)
        for label in self.labels:
            path = os.path.join(self.dir_path, label)
            images = os.listdir(path)
            self.images_dict[label] = images
            for image_id in images:
                img_path = os.path.join(path, image_id)
                self.id2image[image_id] = self.transform(Image.open(img_path))
                self.imgName2label[image_id] = label
        self.list_of_id2image = list(self.imgName2label)
        self.gen_list2d_triplets()

    def __len__(self):
        return len(self.list_of_id2image)

    def __getitem__(self, idx):
        anchor, positive,negative = self.list_of_triplets[idx]
        return self.id2image[anchor], self.id2image[positive],self.id2image[negative]

    def gen_list2d_triplets(self):
        self.list_of_triplets = []
        for anchor in self.list_of_id2image:
            anchor_label = self.imgName2label[anchor]
            positive = random.sample(self.images_dict[anchor_label], k=1)[0]
            negative = random.sample(self.list_of_id2image,k=1)[0]
            while True:
                if positive == anchor:
                    positive = random.sample(self.images_dict[anchor_label], k=1)[0]
                elif self.imgName2label[negative] == anchor_label:
                    negative = random.sample(self.list_of_id2image, k=1)[0]
                else:
                    break
            self.list_of_triplets.append([anchor,positive,negative])
        return self.list_of_triplets


