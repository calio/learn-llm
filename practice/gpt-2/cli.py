#!/usr/bin/env python3
"""
Interactive CLI for GPT-2 model inference.
Load a trained checkpoint and provide an interactive shell for text completion.
"""

import os
import sys
import argparse
import torch
import tiktoken
from typing import Optional

# Add current directory to path to import model
sys.path.append(os.path.dirname(__file__))
from model import GPT2, GPT2Config

class ModelCLI:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Initialize the interactive CLI with a trained model.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._get_device(device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model = None
        
        print(f"🚀 GPT-2 Interactive CLI")
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        print(f"🖥️  Device: {self.device}")
        
        self._load_model()
        print(f"✅ Model loaded successfully!")
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
    
    def _load_model(self):
        """Load the model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
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
        
        # Print checkpoint info
        step = checkpoint.get('step', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"📈 Checkpoint info: Step {step}, Epoch {epoch}")
    
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
            # Tokenize the prompt
            if prompt.strip():
                input_ids = self.tokenizer.encode(prompt)
            else:
                # Empty prompt starts with EOT token
                input_ids = [self.tokenizer.eot_token]
            
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
            generated_text = self.tokenizer.decode(generated[0].tolist())
            
            # Return only the generated part (after the prompt)
            if prompt.strip():
                # Find where the new generation starts
                try:
                    prompt_end = generated_text.find(prompt) + len(prompt)
                    return generated_text[prompt_end:]
                except:
                    # Fallback: return everything after the prompt tokens
                    generated_only = generated[0][len(input_ids):].tolist()
                    return self.tokenizer.decode(generated_only)
            else:
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
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                       help="Path to the .pt checkpoint file")
    parser.add_argument("-d", "--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to run inference on")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Default maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Default sampling temperature")
    parser.add_argument("--top-k", type=int, default=40,
                       help="Default top-k value (0 to disable)")
    
    args = parser.parse_args()
    
    try:
        # Create and run CLI
        cli = ModelCLI(args.checkpoint, args.device)
        cli.run_interactive()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
