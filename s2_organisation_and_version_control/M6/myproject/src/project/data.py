from pathlib import Path

import typer
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

        train_images , train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{raw_data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{raw_data_path}/train_target_{i}.pt"))
        self.train_images = torch.cat(train_images)
        self.train_target = torch.cat(train_target)


        self.test_images: torch.Tensor = torch.load(f"{raw_data_path}/test_images.pt")
        self.test_target: torch.Tensor = torch.load(f"{raw_data_path}/test_target.pt")


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.test_images)+len(self.train_images)                

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if index < len(self.train_images):
            return self.train_images[index]
        else:
            return self.test_images[index-len(self.test_images)]

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        return (images - images.mean()) / images.std()

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        train_images = self.train_images.unsqueeze(1).float()
        test_images = self.test_images.unsqueeze(1).float()
        train_target = self.train_target.long()
        test_target = self.test_target.long()

        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        torch.save(train_images, f"{output_folder}/train_images.pt")
        torch.save(train_target, f"{output_folder}/train_target.pt")
        torch.save(test_images, f"{output_folder}/test_images.pt")
        torch.save(test_target, f"{output_folder}/test_target.pt")        



def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess( output_folder)


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
