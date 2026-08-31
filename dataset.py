import os
import requests

import tokenizers
import torch


DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
}
VOCAB_SIZE = 10000
BATCH_SIZE = 32
TOKENIZER_FILENAME = "gutenberg_tokenizer.json"

# Write dataset files to disc, if they don't exist already
for k,v in DATASOURCE.items():
    filename = f"{k}.txt"
    if not os.path.exists(filename):
        print(f"Downloading: {k}")
        r = requests.get(v)
        with open(filename, "wb") as f:
            f.write(r.content)


# Read and preprocess text
def preprocess_guttenberg(filename):
    with open(file=filename, mode="r", encoding="utf-8") as f:
        text = f.read()

    # Find start and end of book
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract text contents
    text = text[start:end].strip()

    # Basic preprocessing
    # Remove multiple newlines and spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        print(f"Processing: {filename}.txt")
        all_text.append(preprocess_guttenberg(f"{filename}.txt"))
    return all_text

def get_tokenizer():
    # Create Byte-Pair Encoding tokenizer
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decode = tokenizers.decoders.ByteLevel()
    return tokenizer

def train_tokenizer(dataset, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
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

class GuttenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, tokenizer, seq_len=512):
        self.seq_len = seq_len
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

def create_dataset() -> GuttenbergDataset:
    dataset_text_list = get_dataset_text()
    # print(len(dataset_text))
    # Train tokenizer on list data
    if os.path.exists(TOKENIZER_FILENAME):
        tokenizer = tokenizers.Tokenizer.from_file(TOKENIZER_FILENAME)
    else:
        tokenizer = train_tokenizer(dataset_text_list)
    dataset_text_str = "\n".join(dataset_text_list)
    dataset = GuttenbergDataset(dataset_text_str, tokenizer)
    return dataset

if __name__ == "__main__":
    dataset = create_dataset()
    print(len(dataset))
    x, y = dataset[0]
    print(x)
    print(y)
