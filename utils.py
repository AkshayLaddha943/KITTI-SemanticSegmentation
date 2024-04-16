import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import network
import numpy as np
from encoding import *

# def rgb_encoding_to_segmentation_map(img, rgb_to_classid):
#     img = img.permute(1,2,0)
#     segmentation_map = torch.zeros([img.shape[0], img.shape[1]], dtype=torch.uint8)
    
#     for color, class_id in rgb_to_classid.items():
#         condition = torch.all(torch.eq(img, torch.tensor(color, dtype=torch.uint8)), dim=-1)
#         segmentation_map = torch.where(condition, torch.tensor(class_id, dtype=torch.uint8), segmentation_map)
        
#     # Add dimension to change the shape from [height, width] to [height, width, 1]
#     segmentation_map = segmentation_map.unsqueeze(-1)
#     # print(segmentation_map.shape)
        
#     return segmentation_map

def convert_rgb_encoding_to_segmentation_map(image, rgb_to_class_id):

    segmentation_map = tf.zeros([image.shape[0], image.shape[1]], dtype=tf.uint8)

    for color, class_id in rgb_to_class_id.items():
        segmentation_map = tf.where(
                                    condition=tf.reduce_all(tf.equal(image, color), axis=-1),
                                    x=tf.cast(class_id, tf.uint8),
                                    y=segmentation_map
                                    )    
    return segmentation_map

def convert_segmentation_map_to_rgb_encoding(segment_map, rgb_to_class_id):

    rgb_map = np.zeros([segment_map.shape[0], segment_map.shape[1], 3], dtype=np.uint8)

    for color, class_id in rgb_to_class_id.items():
        rgb_map[segment_map==class_id] = color
        
    return rgb_map

def create_mask(pred_mask):
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.cpu().squeeze(0)
    return pred_mask

def predict_mask_pix(dataset, num_image):
    t = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]
                                 )
                             ])
    img, mask = dataset[num_image]
    img = t(img)
    mask = torch.Tensor(mask)
    img, mask = img.cuda(), mask.cuda()
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    output = network(img)
    pred_mask = create_mask(output)
    return img.cpu(), mask.cpu(), pred_mask


img = tf.image.decode_image(tf.io.read_file("C:/KITTI-Segmentation/kitti_data/datasets/kitti/semantic_rgb/000001_10.png"), channels=3)
img = tf.image.resize(img, [375, 1242], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
img = convert_rgb_encoding_to_segmentation_map(img, rgb_to_class_id)
y, idx = tf.unique(tf.reshape(img, -1))
print(y.numpy())


plt.imshow(img)
plt.title("Segmentation Map")
plt.show()