from face_detection.utils.FDDB import preprocess_image
from typing import Generator, Iterable
from numpy import random
from PIL import Image
import numpy as np
import json

import torch

from face_detection.utils.transform import safe_to_tensor
from face_detection import cig

# quick start dataset for my training
class Dataset(object):
    def __init__(self, meta_path : str = "./data/meta.json", train_ratio : float = 0.8) -> None:
        """
            a wrapper for date loader, you can get a easy use loader for the training and test
        Args:
            - meta_path(str) : path of the meta file, please ensure the meta file is a json
            - train_ratio(float) : ratio of the training set
        """
        super().__init__()
        with open(meta_path, "r", encoding="utf-8") as fp:
            self.meta_data : dict = json.load(fp)

        # basic information
        self.train_ratio = train_ratio
        self.offline = int(self.sample_num() * train_ratio)
        self.img_paths = list(self.meta_data.keys())

        # shuffle and split the dataset into training set and test set
        random.shuffle(self.img_paths)
        self.__split_indices()

    def sample_num(self):
        return len(self.meta_data)
    
    def training_sample_num(self):
        return len(self.training_indices)
    
    def test_sample_num(self):
        return len(self.test_indices)
    
    def __split_indices(self):
        self.training_indices : np.ndarray = np.arange(0, self.offline)
        self.test_indices : np.ndarray = np.arange(self.offline, self.sample_num())
    
    # return a generator
    def __abstract_loader(self, sample_indices : Iterable, batch_size : int = cig.batch_size, shuffle : bool = True) -> Generator:
        batch_img = []
        batch_bbox = []
        batch_label = []
        scales = []
        indices = np.array(sample_indices)
        if shuffle:
            np.random.shuffle(indices)
        for index in indices:
            img : np.ndarray = np.array(Image.open(self.img_paths[index]))
            
            img = img.transpose([2, 0, 1])
            origin_height = img.shape[1]

            # remember to preprocess the image and calculate the scale
            img = preprocess_image(img, min_size=cig.min_size, max_size=cig.max_size)
            cur_height = img.shape[1]
            scale = cur_height / origin_height

            bbox = self.meta_data[self.img_paths[index]]
            batch_img.append(img)
            batch_bbox.append(bbox)

            # NOTE: if you are doing a multi-class, change the label generator below
            batch_label.append(np.zeros(len(bbox)))
            scales.append(scale)

            if len(batch_img) == batch_size:
                yield [
                    torch.FloatTensor(batch_img),
                    torch.FloatTensor(batch_bbox),
                    torch.LongTensor(batch_label),
                    scales
                ]
                batch_img.clear()
                batch_bbox.clear()
                batch_label.clear()
                scales.clear()

        batch_img.clear()
        batch_bbox.clear()
        batch_label.clear()
        scales.clear()
    
    def get_loader(self, batch_size : int = cig.batch_size, shuffle : bool = True) -> Generator:
        return self.__abstract_loader(
            sample_indices=np.arange(0, self.sample_num()),
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def get_train_loader(self, batch_size : int = cig.batch_size, shuffle : bool = True) -> Generator:
        return self.__abstract_loader(
            sample_indices=self.training_indices,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def get_test_loader(self, batch_size : int = cig.batch_size, shuffle : bool = True) -> Generator:
        return self.__abstract_loader(
            sample_indices=self.test_indices,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def __len__(self):
        return self.sample_num(self.sample_num())
