import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SRDataset(Dataset):
    """Dataset for the SR GAN, prepares high-res and low-res image pairs for training."""

    def __init__(self, image_dir, hr_image_size):
        """
        Initializes the dataset.
        Args:
            image_dir: The path to the directory containing high resolution images.
            hr_image_size: Integer, the crop size of the images to train on (High
                           resolution images will be cropped to this width and height).
        """
        self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(('jpg', 'jpeg', 'png'))]
        self.image_size = hr_image_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor()
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if image.size[0] < self.image_size or image.size[1] < self.image_size:
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        high_res = self.transform(image)
        low_res = transforms.functional.resize(high_res, (self.image_size // 4, self.image_size // 4), Image.BICUBIC)
        
        # Rescale the pixel values to the -1 to 1 range
        high_res = high_res * 2.0 - 1.0

        return low_res, high_res

def get_dataloader(image_dir, hr_image_size, batch_size, num_workers=4):
    """
    Returns a PyTorch DataLoader with specified mappings.
    Args:
        image_dir: The path to the directory containing high resolution images.
        hr_image_size: Integer, the crop size of the images to train on.
        batch_size: Int, The number of elements in a batch returned by the dataloader.
        num_workers: Int, CPU threads to use for multi-threaded operation.
    Returns:
        dataloader: A PyTorch DataLoader object.
    """
    dataset = SRDataset(image_dir, hr_image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    return dataloader
