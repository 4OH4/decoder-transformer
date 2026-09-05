import os
import requests

import tokenizers
import torch


VOCAB_SIZE = 10
SEQ_LEN = 512
TOKENIZER_FILENAME = "toy_tokenizer.json"


def get_dataset_text():
    # A toy dataset for exercising the transformer
    text = "1011001111000011111111000000001111111111111111000000000000000010110011110000111111110000000011111111111111110000000000000000101100111100001111111100000000111111111111111100000000000000001011001111000011111111000000001111111111111111000000000000000010110011110000111111110000000011111111111111110000000000000000101100111100001111111100000000111111111111111100000000000000001011001111000011111111000000001111111111111111000000000000000010110011110000111111110000000011111111111111110000000000000000"
    return [text] * 8

def create_tokenizer() -> tokenizers.Tokenizer:
    # Create Byte-Pair Encoding tokenizer
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    return tokenizer

def train_tokenizer(dataset:list , tokenizer=None) -> tokenizers.Tokenizer:
    if tokenizer is None:
        tokenizer = create_tokenizer()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens = ["[pad]", "[eos]"],
        show_progress = True
    )
    print("Training tokenizer...")
    tokenizer.train_from_iterator(dataset, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    tokenizer.save(TOKENIZER_FILENAME, pretty=True)
    print("Tokenizer saved")
    return tokenizer

class ToyBinaryDataset(torch.utils.data.Dataset):
    tokenizer:  tokenizers.Tokenizer
    seq_len: int

    def __init__(self, text: str, tokenizer: tokenizers.Tokenizer, seq_len=SEQ_LEN):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

def get_tokenizer(dataset_text: list = None) -> tokenizers.Tokenizer:
    tokenizer = None
    if os.path.exists(TOKENIZER_FILENAME):
        tokenizer = load_tokenizer(TOKENIZER_FILENAME)
    else:
        # Train tokenizer (on list data)
        if dataset_text is not None:
            tokenizer = train_tokenizer(dataset_text)
    if tokenizer is None:
        raise Exception("Could not initialise tokenizer: No saved tokenizer found and no training dataset provided.")
    return tokenizer

def load_tokenizer(filename=TOKENIZER_FILENAME) -> tokenizers.Tokenizer:
    if os.path.exists(filename):
        return tokenizers.Tokenizer.from_file(filename)
    else:
        raise Exception("No trained tokenizer found - run train.py")

def create_dataset() -> ToyBinaryDataset:
    dataset_text_list = get_dataset_text()
    # print(len(dataset_text))    
    tokenizer = get_tokenizer(dataset_text_list)
    dataset_text_str = "\n".join(dataset_text_list)
    dataset = ToyBinaryDataset(dataset_text_str, tokenizer)
    return dataset

if __name__ == "__main__":
    dataset = create_dataset()
    print(len(dataset))
    x, y = dataset[0]
    print(x)
    print(y)
