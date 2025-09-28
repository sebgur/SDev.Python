import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """ A Dataset is an object that takes a piece of text and creates the input and target
        chunks out of it, to be used for training.

        Note that the target chunk is always the result of shifting the input chunk by 1 token.

        The stride is by how many tokens we shift along the sequence to create the next input chunk.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)
        self.text_size_ = len(token_ids)

        # Sliding windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def text_size(self):
        return self.text_size_


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """ A DataLoader is an object that takes a Dataset and construct batches (lists) of tensors of 
        length 2 each containing a pair made out of the input and its target window. """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)

    # return dataset, dataloader # If we want to see the dataset object
    return dataloader

if __name__ == "__main__":
    file = "datasets/llms/the-verdict.txt"
    with open(file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # The targets are always the inputs shifted by 1 token
    # STRIDE: how many tokens we shift to create the next input chunk
    # MAX_LENGTH: context length
    # BATCH_SIZE: how many pairs (input, target) get loaded at each iteration of the DataLoader

    BATCH_SIZE = 5
    MAX_LENGTH = 4
    STRIDE = 3  # When STRIDE = MAX_LENGTH, the next input start right after the previous
    dataloader = create_dataloader_v1(raw_text, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE,
                                      shuffle=False)

    data_iter = iter(dataloader)

    # A 'batch' returned by the DataLoader is effectively a list of dimension 2 where each item is
    # a tensor. The first item is the tensor of inputs and the second item is the tensor of targets.

    # print(f"Text size: {dataset.text_size()}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    print(f"STRIDE: {STRIDE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    # print(f"Batch 1st dimension: {len(first_batch)}")

    inputs, targets = next(data_iter)
    print("Inputs\n", inputs)
    print("Targets\n", targets)
