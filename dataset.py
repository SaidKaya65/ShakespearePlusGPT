"""
Shakespeare Dataset Module

This module handles loading and preparing the Tiny Shakespeare dataset.
We use character-level tokenization for simplicity.

Key Concepts:
1. Tokenization - Converting characters to numbers
2. Train/Validation Split - Separating data for training and evaluation
3. DataLoader - Efficient batching for training
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

# URL for the Tiny Shakespeare dataset (hosted by Andrej Karpathy)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_shakespeare():
    """Download the Shakespeare dataset if not already present."""
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Tiny Shakespeare dataset...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")


class CharacterTokenizer:
    """
    Character-level Tokenizer

    Converts text to numbers and back.
    Each unique character gets a unique ID.

    Example:
        'hello' -> [7, 4, 11, 11, 14]
        [7, 4, 11, 11, 14] -> 'hello'
    """

    def __init__(self, text: str):
        # 1. Get all unique characters and sort them
        self.characters = sorted(list(set(text)))

        # 2. Store vocabulary size
        self.vocab_size = len(self.characters)

        # 3. Create character to ID mapping
        self.char_to_id = {}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        # 4. Create ID to character mapping
        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        # 5. Print vocabulary info
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list:
        """Convert text to list of integer IDs."""
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        """Convert list of integer IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return ''.join(characters)


class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Shakespeare text.

    For language modeling, we create input-target pairs:
    - Input:  [char_0, char_1, char_2, ..., char_n-1]
    - Target: [char_1, char_2, char_3, ..., char_n]

    The model learns to predict the next character at each position.
    """

    def __init__(self,
                 data: torch.Tensor,
                 block_size: int):
        """
        Args:
            data: Tensor of token IDs
            block_size: Length of each training sequence
        """
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of possible starting positions
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple:
        # 1. Get a chunk of data (block_size + 1 for input and target)
        chunk = self.data[index: index + self.block_size + 1]

        # 2. Input is all characters except the last
        x = chunk[:-1]

        # 3. Target is all characters except the first (shifted by 1)
        y = chunk[1:]

        return x, y


def create_dataloaders(batch_size: int = 64,
                       block_size: int = 256,
                       train_split: float = 0.9) -> tuple:
    """
    Create training and validation DataLoaders.

    Args:
        batch_size: Number of sequences per batch
        block_size: Length of each sequence
        train_split: Fraction of data for training (rest is validation)

    Returns:
        train_dataloader, val_dataloader, tokenizer
    """
    # 1. Download the dataset
    download_shakespeare()

    # 2. Load the text file
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    # 3. Create the tokenizer
    tokenizer = CharacterTokenizer(text)

    # 4. Encode the entire text to token IDs
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    # 5. Split into training and validation sets
    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"\nTrain size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    # 6. Create Dataset objects
    train_dataset = ShakespeareDataset(data=train_data, block_size=block_size)
    val_dataset = ShakespeareDataset(data=val_data, block_size=block_size)

    # 7. Create DataLoader objects
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"\nTrain batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")

    return train_dataloader, val_dataloader, tokenizer


# Test the dataset
if __name__ == "__main__":
    # 1. Create dataloaders
    train_dataloader, val_dataloader, tokenizer = create_dataloaders(
        batch_size=4,
        block_size=128
    )

    # 2. Get a sample batch
    x, y = next(iter(train_dataloader))

    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    # 3. Decode and show a sample
    print(f"\nSample sequence:")
    print(tokenizer.decode(x[0]))
