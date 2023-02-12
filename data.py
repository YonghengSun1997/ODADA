import numpy as np
import cv2
from PIL import Image
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PrepareDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            self.image_names = os.listdir(osp.join(self.root_dir, "train/images"))
            self.train_data = []
            self.train_labels = []
            
            for f in tqdm(os.listdir(self.root_dir+'/train/images')):
                # train_img = Image.open(self.root_dir+'/train/images/'+f)
                # train_img = train_img.convert("RGB")
                # train_img.save( 'temp.jpg')
                train_img = cv2.imread(self.root_dir+'/train/images/'+f)
                self.train_data.append(train_img)
#                print(train_img)
                # target_img = Image.open(self.root_dir+'/train/masks_selec/'+ f[:f.rindex('.')]+'_mask.png')
                # target_img = target_img.convert("RGB")
                # target_img.save( 'temp.jpg')
                target_img = cv2.imread(self.root_dir+'/train/masks/'+ f[:f.rindex('.')]+'_mask.png')
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
                self.train_labels.append(target_img)

        # else:
            # self.image_names = os.listdir(osp.join(self.root_dir, "test/images"))
            # self.test_data = []
            # for f in tqdm(os.listdir(self.root_dir+'/test/images')):
                # test_img = Image.open(self.root_dir+'/test/images/'+f)
                # test_img = test_img.convert("RGB")
                # test_img.save( 'temp.jpg')
                # test_img = cv2.imread( 'temp.jpg' )
                # self.test_data.append(test_img)
        else:
            self.image_names = os.listdir(osp.join(self.root_dir, "test/images"))
            self.train_data = []
            self.train_labels = []
            
            for f in tqdm(os.listdir(self.root_dir+'/test/images')):
                # train_img = Image.open(self.root_dir+'/test/images/'+f)
                # train_img = train_img.convert("RGB")
                # train_img.save( 'temp.jpg')
                train_img = cv2.imread(self.root_dir+'/test/images/'+f)
                self.train_data.append(train_img)
#                print(train_img)
                # target_img = Image.open(self.root_dir+'/test/masks/'+ f[:f.rindex('.')]+'_mask.png')
                # target_img = target_img.convert("RGB")
                # target_img.save( 'temp.jpg')
                target_img = cv2.imread(self.root_dir+'/test/masks/'+ f[:f.rindex('.')]+'_mask.png')
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
                self.train_labels.append(target_img)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        # if self.train:
        image, mask = self.train_data[item], self.train_labels[item]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            a = mask
            a1 = np.where(a >= 0.5, a, 0.0)
            a2 = np.where(a < 0.5, a1, 1.0)
            mask = a2

        return image, mask
        # else:
            # image = self.test_data[item]

            # if self.transform:
                # image = self.transform(image)

            # return image

    def _check_exists(self):
        return osp.exists(osp.join(self.root_dir, "train")) and osp.exists(osp.join(self.root_dir, "test"))


class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        return cv2.resize(image, self.output_size, cv2.INTER_AREA)


class Normalize:
    def __call__(self, image):
        image = image.astype(np.float32) / 255
        return image


class ToTensor:
    def __call__(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            data = data.transpose((2, 0, 1))
        else:
            print("Unsupported shape!")
        return torch.from_numpy(data)


if __name__ == "__main__":
    prepared_dataset = PrepareDataset(root_dir="./data",
                                     train=True,
                                     transform=transforms.Compose([Rescale(256)]),
                                     target_transform=transforms.Compose([Rescale(256)]))

    for i in range(len(prepared_dataset)):
        image, mask = prepared_dataset[i]

        print(i, image.shape, mask.shape)
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i == 5:
            break
