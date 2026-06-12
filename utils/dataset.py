import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Custom Dataset wrapper
class CIFAR10Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data      = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['img']
        label = self.data[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(batch_size=64):
    hf_data = load_dataset("uoft-cs/cifar10")

    train_dataset = CIFAR10Dataset(hf_data['train'], transform=train_transform)
    test_dataset  = CIFAR10Dataset(hf_data['test'],  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader