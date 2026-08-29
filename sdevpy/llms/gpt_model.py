import gc, logging
from pathlib import Path
import torch, tiktoken
from sdevpy.llms.local_model import LocalModel
from sdevpy.llms.gpt import gpt
from sdevpy.llms.gpt import textgen as tg
from sdevpy.llms.gpt import datasetloader as dsl
from sdevpy.llms.gpt.training import train_gpt_model
log = logging.getLogger(__name__)


class GptModel(LocalModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def respond_prompt(self, prompt: str, **kwargs) -> str:
        """ Single prompt response from model """
        max_tokens = kwargs.get('max_tokens', 4096)
        max_sentences = kwargs.get('max_sentences', 2)
        top_k = kwargs.get('top_k', 15)
        temperature = kwargs.get('temperature', 1.5)

        # Token generator
        token_gen = tg.NextTokenGenerator(top_k=top_k, temperature=temperature)

        # Initialize chat generator
        chat_gen = tg.ChatGenerator(self.device, self.model, self.tokenizer, self.ctx_length, token_gen,
                                    max_new_tokens=max_tokens, max_sentences=max_sentences)

        # Respond
        start_text = prompt
        end_text = chat_gen.end_text(start_text)
        return tg.format_answer(start_text, end_text)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """ Chat-oriented structured response to messages """
        # The expected behaviour of this function will have to be decided in the future.
        # For now we just write back the answer to the concatenated list of inputs in
        # the role/content format.
        prompt = ''
        for message in messages:
            new_prompt = message.get("content", "")
            prompt += new_prompt + "\n"

        return self.respond_prompt(prompt, **kwargs)

    def train(self, text_data: str, **kwargs) -> None:
        """ Train weights on given text """
        train_ratio = kwargs.get('train_ratio', 0.90)
        batch_size = kwargs.get('batch_size', 2)
        epochs = kwargs.get('epochs', 1)
        init_lr = kwargs.get('init_lr', 0.0004)
        start_text = kwargs.get('test_prompt', "Why is the sky blue?")

        # Fix seed for reproducibility
        torch.manual_seed(123)

        log.info("Create dataset loaders")
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        train_loader = dsl.create_dataloader(train_data, batch_size=batch_size, max_length=self.ctx_length,
                                             stride=self.ctx_length, drop_last=True, shuffle=True, num_workers=0)

        val_loader = dsl.create_dataloader(val_data, batch_size=batch_size, max_length=self.ctx_length,
                                           stride=self.ctx_length, drop_last=False, shuffle=False, num_workers=0)

        log.info("<><><><><><><><> Start training <><><><><><><><>")
        log.info("Test prompt: " + start_text)

        # Initialize model
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr, weight_decay=0.1)

        train_losses, val_losses, tokens_seen = train_gpt_model(self.model, train_loader, val_loader, optimizer,
                                                                self.device, num_epochs=epochs, eval_freq=5,
                                                                eval_iter=5, start_context=start_text,
                                                                tokenizer=self.tokenizer)

    def pretty_print(self) -> None: # pragma: no cov
        """ Display information about the GPT model """
        model_config = self.model.config
        print(f"Model path: {self.path}")
        print(f"Model name: {self.config.get('name', 'Unknown')}")
        print("Architecture: GPT2")
        print(f"Context length: {self.ctx_length}")
        print(f"Attention heads: {model_config.get('n_heads', 'Unknown')}")
        print(f"Transformer blocks: {model_config.get('n_layers', 'Unknown')}")
        print(f"Embedding dimension: {model_config.get('emb_dim', 'Unknown')}")
        print(f"Context length: {model_config.get('vocab_size', 'Unknown')}")

    def load(self, max_context_tokens: int=None, device=None) -> None:
        """ Load GPT model from saved weights and config """
        path = Path(self.config.get("path"))
        if not path.exists():
            raise ValueError(f"Model data path not found: {path}")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = gpt.load_model_from_path(path, self.device)
        self.path = path
        self.ctx_length = self.model.config.get('context_length', None)
        if self.ctx_length is None:
            raise ValueError("No context length found in GPT model config")

    def unload(self) -> None:
        """ Unload Llama model from memory """
        del self.model
        gc.collect()


if __name__ == "__main__":
    from sdevpy.tests import conftest as tst
    test_path = tst.calibdata_path() / "gpt" / "gpt2-test"
    repo_config = {"type": "gpt", "name": "gpt2-test", "path": test_path}
    print(test_path)

    data_file = tst.dataset_path() / "llms" / "the-verdict.txt"
    with open(data_file, encoding="utf-8") as f:
        text_data = f.read()

    # print(text_data)

    model = GptModel(repo_config)
    model.load()

    model.train(text_data)

    print("Testing pretrained")
    print(model.respond_prompt("Why is the sky green?"))

    model.unload()

