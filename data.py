import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from encoding import rgb_to_class_id
from utils import convert_rgb_encoding_to_segmentation_map

import tensorflow as tf

import cv2
import os
import glob

T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(
                                 [0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]
                                 )
                             ])

class KittiDataset(Dataset):
    def __init__(self, image_paths, segment_paths):
        self.image_paths = image_paths
        self.segment_paths = segment_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.resize(image, (256,256))

        label = cv2.imread(self.segment_paths[index])
        label = cv2.resize(label, (256,256))
        label = convert_rgb_encoding_to_segmentation_map(label, rgb_to_class_id)
        label = label.numpy()

        return image, label

    def collate_fn(self, batch):
      ims, labels = list(zip(*batch))
      ims = torch.cat([T(im.copy()/255.)[None] for im in ims]).float()
      ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in labels]).long()
      return ims, ce_masks


image_path = sorted(glob.glob(os.path.join("C:/KITTI-Segmentation/kitti_data/datasets/kitti/image_2/", "*.png")))
segment_path = sorted(glob.glob(os.path.join("C:/KITTI-Segmentation/kitti_data/datasets/kitti/semantic_rgb/", "*.png")))
batch_size = 8

print(image_path[4])
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_path, segment_path, train_size=0.7, random_state=0)

# Keep part of the validation set as test set
val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(val_image_paths, val_mask_paths, train_size = 0.75, random_state=0)

train_dataset = KittiDataset(train_image_paths, train_mask_paths)
val_dataset = KittiDataset(val_image_paths, val_mask_paths)
test_dataset = KittiDataset(test_image_paths, test_mask_paths)

print(f'There are {train_dataset.__len__()} images in the Training Set')
print(f'There are {val_dataset.__len__()} images in the Validation Set')
print(f'There are {test_dataset.__len__()} images in the Validation Set')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)