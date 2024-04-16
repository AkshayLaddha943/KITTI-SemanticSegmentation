import torch.nn as nn
import torch

def metrics(model):
  unet_loss = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  return unet_loss, optim