import logging
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sdevpy.llms.local_model import LocalModel


class TransformersModel(LocalModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.tokenizer = None

    def respond_prompt(self, prompt: str, **kwargs) -> str:
        """ Single prompt response from underlying Transformers model """
        max_tokens = kwargs.get('max_tokens', 4096)
        temperature = kwargs.get('temperature', 0.2)
        num_beams = kwargs.get('num_beams', 1)
        do_sample = kwargs.get('do_sample', True)
        top_p = kwargs.get('top_p', 0.9)

        # Create prompt and tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            response = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                           num_beams=num_beams, do_sample=do_sample, top_p=top_p,
                                           pad_token_id=self.tokenizer.eos_token_id)

        # Decode response
        response = self.tokenizer.decode(response[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)

        # Return clean response
        return self.clean_response(response)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """ Chat-oriented response to messages """
        max_tokens = kwargs.get('max_tokens', 4096)
        temperature = kwargs.get('temperature', 0.2)
        num_beams = kwargs.get('num_beams', 1)
        do_sample = kwargs.get('do_sample', True)
        top_p = kwargs.get('top_p', 0.9)

        if not self.tokenizer.chat_template:
            raise ValueError("Model does not have a chat template: use an instruct model")

        # Create prompt and tokenize
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            response = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                           num_beams=num_beams, do_sample=do_sample, top_p=top_p,
                                           pad_token_id=self.tokenizer.eos_token_id)

        # Decode response
        response = self.tokenizer.decode(response[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Return clean response
        return self.clean_response(response)

    def pretty_print(self) -> None:
        """ Display information about the model """
        print(f"Model name: {self.model.config.name_or_path}")
        print(f"Model type: {self.model.config.model_type}")
        print(f"Architecture: {self.model.config.architectures}")

        # Number parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {total_params/1e9:.3f}B")
        # print(f"Number of parameters: {self.model.num_parameters():,}") # Should match the above
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params/1e9:.3f}B")

        # Device
        print(f"Running on: {self.model.device}")

        # Size in memory
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        size_gb = total_size / (1024**3) # Convert to GB
        print(f"Size(GB): {size_gb:.3f}")

    def load(self) -> None:
        """ Load Transformers model from Hugging Face into memory """
        hf_repo_id = self.config.get("repo_id")

        logger = logging.getLogger("transformers.modeling_utils")
        original_level = logger.level
        logger.setLevel(logging.ERROR)

        self.model = AutoModelForCausalLM.from_pretrained(hf_repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)

        logger.setLevel(original_level)

    def unload(self) -> None:
        """ Unload Transformers model from memory """
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
