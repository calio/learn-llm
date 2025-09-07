#!/usr/bin/env python3
"""
Interactive CLI for GPT-2 model inference.
Load a trained checkpoint or HuggingFace model and provide an interactive shell for text completion.
"""

import os
import sys
import argparse
import torch
import tiktoken
from typing import Optional, Union

# Add current directory to path to import model
sys.path.append(os.path.dirname(__file__))
try:
    from model import GPT2, GPT2Config
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    print("Make sure model.py is in the same directory as cli.py")
    sys.exit(1)

# Try to import HuggingFace transformers (optional)
try:
    from transformers import GPT2LMHeadModel, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  HuggingFace transformers not available. Only local checkpoints supported.")
    HF_AVAILABLE = False

class ModelCLI:
    def __init__(self, model_path: str, device: str = "auto", model_type: str = "auto"):
        """
        Initialize the interactive CLI with a trained model.
        
        Args:
            model_path: Path to checkpoint file or HuggingFace model name/path
            device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps')
            model_type: Type of model ('auto', 'local', 'hf')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model_type = self._detect_model_type(model_path, model_type)
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 GPT-2 Interactive CLI")
        print(f"📁 Model: {model_path}")
        print(f"📝 Type: {self.model_type}")
        print(f"🖥️  Device: {self.device}")
        
        self._load_model()
        print(f"✅ Model loaded successfully!")
        if hasattr(self.model, 'parameters'):
            print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
    
    def _get_device(self, device_arg: str) -> torch.device:
        """Determine the best device to use."""
        if device_arg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_arg)
    
    def _detect_model_type(self, model_path: str, model_type: str) -> str:
        """Detect whether this is a local checkpoint or HuggingFace model."""
        if model_type != "auto":
            return model_type
        
        # Check if it's a local file
        if os.path.exists(model_path) and model_path.endswith('.pt'):
            return "local"
        
        # Check if it's a HuggingFace model (directory with config.json or model name)
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
            return "hf"
        
        # Assume it's a HuggingFace model name if HF is available
        if HF_AVAILABLE:
            return "hf"
        
        # Default to local
        return "local"
    
    def _load_model(self):
        """Load the model from checkpoint or HuggingFace."""
        if self.model_type == "local":
            self._load_local_checkpoint()
        elif self.model_type == "hf":
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_local_checkpoint(self):
        """Load model from local PyTorch checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Create model with default config
        self.model = GPT2()
        
        # Clean state dict (remove wrapper prefixes)
        state_dict = checkpoint['model_state_dict']
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith('_fsdp_wrapped_module.'):
                clean_key = clean_key.replace('_fsdp_wrapped_module.', '')
            if clean_key.startswith('module.'):
                clean_key = clean_key.replace('module.', '')
            cleaned_state_dict[clean_key] = value
        
        # Load state dict
        self.model.load_state_dict(cleaned_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Use tiktoken for local models
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Print checkpoint info
        step = checkpoint.get('step', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"📈 Checkpoint info: Step {step}, Epoch {epoch}")
    
    def _load_huggingface_model(self):
        """Load model from HuggingFace."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
        
        try:
            # Load model and tokenizer
            print("🤗 Loading HuggingFace model...")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"📈 Model info: {self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'HuggingFace model'}")
            
        except Exception as e:
            raise Exception(f"Failed to load HuggingFace model: {e}")
    
    def _encode_text(self, text: str):
        """Encode text using the appropriate tokenizer."""
        if self.model_type == "local":
            if text.strip():
                return self.tokenizer.encode(text)
            else:
                return [self.tokenizer.eot_token]
        else:  # HuggingFace
            return self.tokenizer.encode(text, return_tensors="pt").squeeze(0).tolist()
    
    def _decode_text(self, token_ids):
        """Decode token IDs using the appropriate tokenizer."""
        if self.model_type == "local":
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return self.tokenizer.decode(token_ids)
        else:  # HuggingFace
            if isinstance(token_ids, list):
                token_ids = torch.tensor(token_ids)
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0, 
                     top_k: Optional[int] = 40) -> str:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = no top-k)
        
        Returns:
            Generated text completion
        """
        with torch.no_grad():
            if self.model_type == "local":
                return self._generate_local(prompt, max_tokens, temperature, top_k)
            else:  # HuggingFace
                return self._generate_huggingface(prompt, max_tokens, temperature, top_k)
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float, top_k: Optional[int]) -> str:
        """Generate text using local model."""
        # Tokenize the prompt
        input_ids = self._encode_text(prompt)
        
        # Convert to tensor
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]
        
        # Generate
        generated = self.model.generate(
            input_tensor, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode the full sequence
        generated_text = self._decode_text(generated[0])
        
        # Return only the generated part (after the prompt)
        if prompt.strip():
            # Find where the new generation starts
            try:
                prompt_end = generated_text.find(prompt) + len(prompt)
                return generated_text[prompt_end:]
            except:
                # Fallback: return everything after the prompt tokens
                generated_only = generated[0][len(input_ids):].tolist()
                return self._decode_text(generated_only)
        else:
            return generated_text
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float, top_k: Optional[int]) -> str:
        """Generate text using HuggingFace model."""
        # Tokenize the prompt
        input_text = prompt if prompt.strip() else self.tokenizer.eos_token
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Set up generation parameters
        generation_kwargs = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        
        if top_k is not None:
            generation_kwargs['top_k'] = top_k
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(inputs, **generation_kwargs)
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(generated[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return generated_text
    
    def run_interactive(self):
        """Run the interactive CLI loop."""
        print("🎯 Interactive mode started!")
        print("📝 Type your prompts below. Commands:")
        print("   • /help - Show this help")
        print("   • /settings - Show current settings") 
        print("   • /temp <value> - Set temperature (e.g., /temp 0.8)")
        print("   • /tokens <value> - Set max tokens (e.g., /tokens 150)")
        print("   • /topk <value> - Set top-k (e.g., /topk 50, /topk none)")
        print("   • /quit or /exit - Exit the CLI")
        print("   • Empty line - Generate from EOT token")
        print()
        
        # Default settings
        temperature = 1.0
        max_tokens = 100
        top_k = 40
        
        while True:
            try:
                # Get user input
                prompt = input("💬 Prompt: ").strip()
                
                # Handle commands
                if prompt.startswith('/'):
                    command_parts = prompt[1:].split()
                    command = command_parts[0].lower()
                    
                    if command in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    elif command == 'help':
                        print("\n📖 Available commands:")
                        print("   /help - Show this help")
                        print("   /settings - Show current settings")
                        print("   /temp <value> - Set temperature")
                        print("   /tokens <value> - Set max tokens") 
                        print("   /topk <value> - Set top-k")
                        print("   /quit, /exit - Exit")
                        print()
                        continue
                    
                    elif command == 'settings':
                        print(f"\n⚙️  Current settings:")
                        print(f"   Model: {self.model_path}")
                        print(f"   Type: {self.model_type}")
                        print(f"   Temperature: {temperature}")
                        print(f"   Max tokens: {max_tokens}")
                        print(f"   Top-k: {top_k}")
                        print(f"   Device: {self.device}")
                        print()
                        continue
                    
                    elif command == 'temp' and len(command_parts) > 1:
                        try:
                            temperature = float(command_parts[1])
                            print(f"🌡️  Temperature set to {temperature}")
                        except ValueError:
                            print("❌ Invalid temperature value. Use a number (e.g., /temp 0.8)")
                        continue
                    
                    elif command == 'tokens' and len(command_parts) > 1:
                        try:
                            max_tokens = int(command_parts[1])
                            print(f"📏 Max tokens set to {max_tokens}")
                        except ValueError:
                            print("❌ Invalid token count. Use an integer (e.g., /tokens 150)")
                        continue
                    
                    elif command == 'topk' and len(command_parts) > 1:
                        try:
                            if command_parts[1].lower() == 'none':
                                top_k = None
                                print("🔝 Top-k disabled")
                            else:
                                top_k = int(command_parts[1])
                                print(f"🔝 Top-k set to {top_k}")
                        except ValueError:
                            print("❌ Invalid top-k value. Use an integer or 'none' (e.g., /topk 50)")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {command}. Type /help for available commands.")
                        continue
                
                # Generate text
                print("🤖 Generating...", end="", flush=True)
                try:
                    generated = self.generate_text(
                        prompt, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                    print(f"\r🤖 Generated: {generated}")
                except Exception as e:
                    print(f"\r❌ Generation failed: {e}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break

def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for GPT-2 model")
    parser.add_argument("model", type=str,
                       help="Path to .pt checkpoint file or HuggingFace model name/path")
    parser.add_argument("-d", "--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to run inference on")
    parser.add_argument("-t", "--type", type=str, default="auto",
                       choices=["auto", "local", "hf"],
                       help="Model type (auto=detect, local=PyTorch checkpoint, hf=HuggingFace)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Default maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Default sampling temperature")
    parser.add_argument("--top-k", type=int, default=40,
                       help="Default top-k value (0 to disable)")
    
    args = parser.parse_args()
    
    try:
        # Create and run CLI
        cli = ModelCLI(args.model, args.device, args.type)
        cli.run_interactive()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
