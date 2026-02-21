import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input(entry):
    """ Prompt style Alpaca """
    instruction_text = (f"Below is an instruction that describes a task. "
                        f"Write a response that appropriately complets the request."
                        f"\n\n### Instruction:\n{entry['instruction']}")

    input_text = (f"\n\n### Input:\n{entry['input']}" if entry["input"] else "")
    return instruction_text + input_text

