
# Basic python imports for logging and sequence generation
import itertools

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dataset of binary strings, during training generates up to length max_length
# During testing, just create sequences of max_length
class Parity(Dataset):

    def __init__(self, training=True, max_length=4, samples=1000):
      super().__init__()
      self.training = training
      self.max_length = max_length
      if self.training:
        self.data = torch.FloatTensor(list(itertools.product([0,1], repeat=max_length)))
      else:
        self.data = torch.randint(low=0, high=2, size=(samples,max_length)).to(dtype=torch.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.training and self.max_length > 1:
           l = torch.randint(low=1, high=self.max_length, size=(1,1)).item()
           x = self.data[idx][:l]
        else:
           x = self.data[idx]

        y = x.sum() % 2
        return x,y


    # Function to enable batch loader to stack binary strings of different lengths and pad them
    @staticmethod
    def pad_collate(batch):
        # Unzip batch into x and y sequences
        x_batch, y_batch = zip(*batch)

        # Convert x_batch to tensors of type float32
        x_seqs = [torch.tensor(x, dtype=torch.float32) for x in x_batch]

        # Convert y_batch to a tensor of type long
        y_seqs = torch.tensor([int(y) for y in y_batch], dtype=torch.long)

        # Calculate the lengths of x sequences
        x_lens = torch.tensor([len(x) for x in x_batch], dtype=torch.long)

        # Pad the x sequences
        x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0)

        return x_padded, y_seqs, x_lens



def getParityDataloader(training=True, max_length=4, batch_size=1000):
  dataset = Parity(training, max_length)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=training, collate_fn=dataset.pad_collate, drop_last=False)
  return loader
