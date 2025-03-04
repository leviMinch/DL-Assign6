import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from datasets.PoSData import Vocab, getUDPOSDataloaders
from models.PoSGRU import PoSGRU

import pickle

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs":256,   # batch size
    "lr":0.0005, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":30,
    "layers": 2,
    "embed_dim":128,
    "hidden_dim":256,
    "residual":True
}


def main():

  # Get dataloaders
  train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

  vocab_size = vocab.lenWords()
  label_size = vocab.lenLabels()

  # Build model
  model = PoSGRU(vocab_size=vocab_size,
                 embed_dim=config["embed_dim"],
                 hidden_dim=config["hidden_dim"],
                 num_layers=config["layers"],
                 output_dim=label_size,
                 residual=config["residual"])
  print(model)

  torch.compile(model)


  # Start model training
  train(model, train_loader, val_loader)




def train(model, train_loader, val_loader):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="UDPOS CS499 A6", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  warmup_epochs = config["max_epoch"]//10
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  # Loss
  ###########################################
  #
  # Q5 TODO Loss
  #
  ###########################################


  # Main training loop with progress bar
  iteration = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    # Set best_val very high for later checkpointing
    best_val = 10000000

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)


      ###########################################
      #
      # Q5 TODO Loss
      #
      ###########################################
      # Reshape `out` from [batch_size, seq_len, num_classes] to [batch_size * seq_len, num_classes]
      out = out.view(-1, 16)

      # Reshape `y` from [batch_size, seq_len] to [batch_size * seq_len]
      y = y.view(-1)

      # Compute loss
      loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens if applicable
      loss = loss_fn(out, y)


      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      ###########################################
      #
      # Q5 TODO Accuracy
      #
      ###########################################
      acc = (torch.argmax(out, dim=1) == y).to(dtype=float).mean()



      wandb.log({"Loss/train": loss.item(), "Acc/train": acc}, step=iteration)
      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)

    ###########################################
    #
    # Q6 TODO Checkpointing
    #
    ###########################################
    #Model Checkpointing
    if val_loss < best_val:
      best_val = val_loss
      torch.save(model.state_dict(), "chkpts/" + run_name + "_epoch " + str(epoch))

      with open(f"chkpts/{run_name}_vocab.pkl", 'wb') as vocab_file:
        pickle.dump(train_loader.dataset.vocab, vocab_file)

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()


def evaluate(model, loader):
  ###########################################
  #
  # Q6 TODO
  # Used code from assignment 4
  ###########################################
  model.eval()

  running_loss = 0
  running_acc = 0
  criterion = torch.nn.CrossEntropyLoss(reduction="sum")

  for x,y, lens in loader:

    x = x.to(device)
    y = y.to(device)
    lens = lens.to(device)

    out = model(x)

    # Reshape `out` from [batch_size, seq_len, num_classes] to [batch_size * seq_len, num_classes]
    out = out.view(-1, out.shape[-1])  # (256 * 75, 16)

    # Reshape `y` from [batch_size, seq_len] to [batch_size * seq_len]
    y = y.view(-1)  # (256 * 75)
    print("y:", y)
    loss = criterion(out,y)

    acc = (torch.argmax(out, dim=1) == y).to(dtype=float).sum()

    running_loss += loss.item()
    running_acc += acc.item()

  nonpad = len(loader.dataset)
  return running_loss/nonpad, running_acc/nonpad

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_UDPOS"
  return run_name


if __name__ == "__main__":
  main()



# Pull request 5

def remove_odds(arr):
  """ Receives a list of integers and removes all odd numbers and
  returns a sorted list (ascending) without making changes to the original list (arr) """

  new_list = []
  for num in arr:
    if num % 2 == 0:
      new_list.append(num)

  new_list.sort()

  return new_list