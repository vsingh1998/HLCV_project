import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

BATCH_SIZE=16
NUM_WORKER=2

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = datasets.CelebA(root='main/datasets/', small=True, download=True)

total_count = len(dataset) 
train_count = int(0.7 * total_count) 
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count))

train_dataset = train_dataset.map(data_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)  
valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER) 
test_dataset_loader  = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKER)

dataloaders = {'train': train_dataset_loader, 'val': valid_dataset_loader, 'test': test_dataset_loader}

print(len(dataloaders['train']))