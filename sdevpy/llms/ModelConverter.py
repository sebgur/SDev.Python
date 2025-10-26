"""
Convert PyTorch .pth model to GGUF format for Ollama
Assumes model is based on a known architecture (Llama, GPT-2, etc.).
[Seb]: this code was created by Claude Sonnet 4.5. Tried with GPT-2 like models as per Raschka,
but conversion fails. The objective was to convert a local GPT-2 model created according to
Raschka into a format usable by Ollama as just another local model.
"""
import torch
import json
import struct
import numpy as np
from pathlib import Path

class PTHToGGUFConverter:
    """ Convert PyTorch model to GGUF format """
    def __init__(self, pth_path: str, architecture: str = "llama"):
        """ Args:
            pth_path: Path to your .pth file
            architecture: Model architecture (llama, gpt2, mistral, etc.) """
        self.pth_path = pth_path
        self.architecture = architecture
        self.checkpoint = None
        self.config = None

    def load_pth(self):
        """Load and inspect the .pth file"""
        print(f"Loading {self.pth_path}...")
        self.checkpoint = torch.load(self.pth_path, map_location='cpu')

        # Determine structure
        if isinstance(self.checkpoint, dict):
            if 'model_state_dict' in self.checkpoint:
                print("✓ Found model_state_dict")
                self.state_dict = self.checkpoint['model_state_dict']
                self.config = self.checkpoint.get('config', {})
            elif 'state_dict' in self.checkpoint:
                print("✓ Found state_dict")
                self.state_dict = self.checkpoint['state_dict']
                self.config = self.checkpoint.get('config', {})
            else:
                print("✓ Assuming checkpoint is the state_dict")
                self.state_dict = self.checkpoint
        else:
            print("✓ Checkpoint is a model object")
            self.state_dict = self.checkpoint.state_dict()

        print(f"Found {len(self.state_dict)} tensors")

        # Print some keys to understand structure
        print("\nFirst 10 keys:")
        for i, key in enumerate(list(self.state_dict.keys())[:10]):
            print(f"  {key}: {self.state_dict[key].shape}")

    def convert_to_huggingface_format(self, output_dir: str):
        """ Convert to Hugging Face format first (easier for GGUF conversion)
            Args:
            output_dir: Directory to save Hugging Face format model """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nConverting to Hugging Face format at {output_dir}...")

        # This part depends heavily on your model architecture
        # You need to map your state_dict keys to HF format
        if self.architecture == "llama":
            self._convert_to_llama_hf(output_path)
        elif self.architecture == "gpt2":
            self._convert_to_gpt2_hf(output_path)
        else:
            raise ValueError(f"Architecture {self.architecture} not supported yet")

    def _convert_to_llama_hf(self, output_path: Path):
        """Convert to Llama Hugging Face format"""
        from transformers import LlamaConfig, LlamaForCausalLM

        # Create config (you need to fill in your model's actual config)
        config = LlamaConfig(
            vocab_size=self.config.get('vocab_size', 32000),
            hidden_size=self.config.get('hidden_size', 4096),
            intermediate_size=self.config.get('intermediate_size', 11008),
            num_hidden_layers=self.config.get('num_layers', 32),
            num_attention_heads=self.config.get('num_heads', 32),
            num_key_value_heads=self.config.get('num_kv_heads', 32),
            max_position_embeddings=self.config.get('max_seq_len', 2048),
        )

        # Create model with this config
        model = LlamaForCausalLM(config)

        # Map your state_dict to HF format
        # This is the tricky part - you need to rename keys
        hf_state_dict = self._map_state_dict_to_hf_llama(self.state_dict)

        # Load the mapped state dict
        model.load_state_dict(hf_state_dict, strict=False)

        # Save in HF format
        model.save_pretrained(output_path)

        # Save tokenizer if you have it
        if 'tokenizer' in self.checkpoint:
            tokenizer = self.checkpoint['tokenizer']
            tokenizer.save_pretrained(output_path)

        print(f"✓ Saved to {output_path}")
        print("  Next step: Convert this to GGUF using llama.cpp")

    def _map_state_dict_to_hf_llama(self, state_dict):
        """
        Map your custom state_dict keys to Hugging Face Llama format

        Example mappings (YOU NEED TO CUSTOMIZE THIS):
        Your format          → HF format
        'embedding.weight'   → 'model.embed_tokens.weight'
        'layer.0.attn.q'     → 'model.layers.0.self_attn.q_proj.weight'
        'layer.0.attn.k'     → 'model.layers.0.self_attn.k_proj.weight'
        'layer.0.attn.v'     → 'model.layers.0.self_attn.v_proj.weight'
        'layer.0.attn.o'     → 'model.layers.0.self_attn.o_proj.weight'
        'layer.0.mlp.w1'     → 'model.layers.0.mlp.gate_proj.weight'
        'layer.0.mlp.w2'     → 'model.layers.0.mlp.down_proj.weight'
        'layer.0.mlp.w3'     → 'model.layers.0.mlp.up_proj.weight'
        'output.weight'      → 'lm_head.weight'
        """

        hf_state_dict = {}

        # Example mapping - YOU MUST CUSTOMIZE THIS
        for key, tensor in state_dict.items():
            # Remove 'model.' prefix if it exists
            new_key = key.replace('model.', '')

            # Add 'model.' prefix for HF format
            if not new_key.startswith('lm_head'):
                new_key = 'model.' + new_key

            hf_state_dict[new_key] = tensor

        return hf_state_dict

    def _convert_to_gpt2_hf(self, output_path: Path):
        """Convert to GPT-2 Hugging Face format"""
        from transformers import GPT2Config, GPT2LMHeadModel

        # Similar process as Llama but for GPT-2
        config = GPT2Config(
            vocab_size=self.config.get('vocab_size', 50257),
            n_positions=self.config.get('n_positions', 1024),
            n_embd=self.config.get('n_embd', 768),
            n_layer=self.config.get('n_layer', 12),
            n_head=self.config.get('n_head', 12),
        )

        model = GPT2LMHeadModel(config)
        hf_state_dict = self._map_state_dict_to_hf_gpt2(self.state_dict)
        model.load_state_dict(hf_state_dict, strict=False)
        model.save_pretrained(output_path)

        print(f"✓ Saved to {output_path}")


def print_model_info(pth_path: str):
    """Utility to inspect your .pth file"""
    print(f"Inspecting {pth_path}...\n")

    checkpoint = torch.load(pth_path, map_location='cpu')

    print("="*60)
    print("CHECKPOINT STRUCTURE")
    print("="*60)

    if isinstance(checkpoint, dict):
        print("Type: Dictionary")
        print(f"Keys: {list(checkpoint.keys())}\n")

        # Look for state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Look for config
        if 'config' in checkpoint:
            print("CONFIG:")
            print(json.dumps(checkpoint['config'], indent=2))
            print()

        # Print state dict info
        print("STATE DICT:")
        print(f"  Total parameters: {len(state_dict)}")
        print(f"  First 15 keys:")
        for i, (key, tensor) in enumerate(list(state_dict.items())[:15]):
            print(f"    {key}: {tensor.shape} ({tensor.dtype})")

        # Try to infer architecture
        print("\nINFERRED ARCHITECTURE:")
        keys = list(state_dict.keys())
        if any('llama' in k.lower() for k in keys):
            print("  Likely: Llama-based")
        elif any('gpt' in k.lower() for k in keys):
            print("  Likely: GPT-based")
        elif any('bert' in k.lower() for k in keys):
            print("  Likely: BERT-based")
        else:
            print("  Unknown - check key names above")

    else:
        print("Type: Model object")
        print(f"Class: {type(checkpoint)}")
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            print(f"Parameters: {len(state_dict)}")


# Example usage
if __name__ == "__main__":
    import sys

    # Path to PyTorch model file to convert
    pth_file = "C:\\temp\\llms\\models\\gpt-0.1.pth"

    print("="*80)
    print("PyTorch to GGUF Converter")
    print("="*80)
    print()

    # Step 1: Inspect your model
    print("STEP 1: Inspect model .pth file")
    print("-"*80)

    try:
        print_model_info(pth_file)
    except FileNotFoundError:
        print(f"File not found: {pth_file}")
        print("\nUsage:")
        print("  1. Replace 'your_model.pth' with your actual file path")
        print("  2. Run this script to inspect your model")
        print("  3. Based on the output, customize the conversion")
        sys.exit(1)

    print("\n" + "="*80)
    print("STEP 2: Convert to Hugging Face format")
    print("-" * 80)
    print("Uncomment the code below after inspecting your model:\n")
    converter = PTHToGGUFConverter(pth_path=pth_file, architecture='gpt2')  # llama, gpt2, etc.

    converter.load_pth()
    converter.convert_to_huggingface_format('./hf_model')

    print("\n" + "="*80)
    print("STEP 3: Convert HF format to GGUF")
    print("-" * 80)
    print("""
    After Step 2, run these commands:

    # Clone llama.cpp if you haven't
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt

    # Convert to GGUF
    python convert-hf-to-gguf.py ../hf_model --outfile model.gguf --outtype f16

    # Quantize (optional, makes it smaller)
    make
    ./quantize model.gguf model-q4.gguf Q4_K_M
    """)
