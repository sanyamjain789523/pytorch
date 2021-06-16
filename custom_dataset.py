import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda availability: {torch.cuda.is_available()}")

