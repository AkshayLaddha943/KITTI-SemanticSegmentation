import time
from tqdm import tqdm

from model import network
from data import *
from encoding import rgb_to_class_id
from utils import convert_segmentation_map_to_rgb_encoding, create_mask, predict_mask_pix
from metric import metrics
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

res = {"train_loss": [], "val_loss": []}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_loss, optim = metrics(network)

print("Begin Training......")
start_time = time.time()

for i in tqdm(range(50)):
  network.train()
  totalTrainLoss = 0
  totalTestLoss = 0
#   trainpixelaccuracy = 0
#   valpixelaccuracy = 0
  
  for j, data in enumerate(tqdm(train_loader)):
    imgs, labels = data
    imgs, labels = imgs.cuda(), labels.cuda()
    print(imgs.shape, labels.shape)
    output = network(imgs)
    loss = unet_loss(output, labels)
    #train_pixel_acc = pixel_acc(output, masks)
    # iou_score = mIoU(output, masks)
    optim.zero_grad()
    loss.backward()
    optim.step()

    totalTrainLoss += loss
    #trainpixelaccuracy += train_pixel_acc

  with torch.no_grad():

    for k, data in enumerate(tqdm(val_loader)):
      imgs, labels = data
      imgs, labels = imgs.cuda(), labels.cuda()
      output = network(imgs)
      loss = unet_loss(output, labels)
      #val_pixel_acc = pixel_acc(output, masks)
      totalTestLoss += loss
      #valpixelaccuracy += val_pixel_acc

  avgTrainLoss = totalTrainLoss/len(train_loader)
  avgValLoss = totalTestLoss/len(val_loader)
  
#   avgTrainAcc = trainpixelaccuracy/len(train_loader)
#   avgValAcc = valpixelaccuracy/len(val_loader)

  res["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
  res["val_loss"].append(avgValLoss.cpu().detach().numpy())
#   res["train_pixel_accuracy"].append(avgTrainAcc.cpu().detach().numpy())
#   res["validation_pixel_accuracy"].append(avgValAcc.cpu().detach().numpy())

  print("[INFO] EPOCH: {}/{}".format(i + 1, 50))
  print("Train loss: {:.6f}, Validation loss: {:.4f}".format(avgTrainLoss, avgValLoss))
  # print("IoU Score: {:.3f}".format(iou_score))

end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	end_time - start_time))

def plot(trainloss, validloss):
    plt.figure(figsize=(10,10))
    plt.plot(range(0,50,1), trainloss, linestyle = 'dashed', color = 'blue', marker = 'o', label="train loss")
    plt.plot(range(0,50,1), validloss, linestyle = 'dashed', color = 'red', marker = 'o', label="Validation loss")
    plt.title("Training vs Validation Loss vs No. of Epochs ")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Training vs Validation Loss")
    plt.legend()
    plt.show()

plot(res["train_loss"], res["val_loss"])

# for i in range(3):
img, mask, pred_mask = predict_mask_pix(test_dataset, 6)
print(pred_mask.shape)
pred_mask_rgb = convert_segmentation_map_to_rgb_encoding(pred_mask, rgb_to_class_id)
print(pred_mask_rgb.shape)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,20))
ax1.imshow(img.squeeze().permute(1,2,0))
ax1.set_title('Picture')

ax2.imshow(mask.squeeze())
ax2.set_title('Ground truth')
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title('UNet Predicted')
ax3.set_axis_off()

ax4.imshow(pred_mask_rgb)
ax4.set_title('UNet Predicted RGB')
ax4.set_axis_off()

plt.show()