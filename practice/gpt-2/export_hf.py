"""
Script to convert GPT2 models from PyTorch checkpoint format to Hugging Face

It can optionally upload to your account on Hugging Face if you have the CLI:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login

Export to a local HF model:
  python export_hf.py --input checkpoint.pt --output output_dir

Export to a local HF model and also push to your account on Hugging Face:
  python export_hf.py --input checkpoint.pt --output output_dir --push --repo_name my-gpt2-model

Example with a trained checkpoint:
  python export_hf.py --input output/run_20240901_120000/checkpoint_iter_5000.pt --output gpt2-shakespeare --push --repo_name gpt2-shakespeare
"""

import torch
import argparse
import os
import sys
from pathlib import Path
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# Import our model classes (assumes this script is in the same directory as model.py)
from model import GPT2 as CustomGPT2, GPT2Config as CustomGPT2Config, block

# -----------------------------------------------------------------------------
# Main conversion function

def convert_checkpoint_to_hf(checkpoint_path, output_dir, push_to_hub=False, repo_name=None, out_dtype="bfloat16"):
    """
    Convert a PyTorch checkpoint to Hugging Face format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save the HF model
        push_to_hub: Whether to push to Hugging Face Hub
        repo_name: Repository name for Hugging Face Hub
        out_dtype: Output dtype (float32 or bfloat16)
    """
    print(f"Converting checkpoint {checkpoint_path} to {output_dir} in {out_dtype} format")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}, epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint (no metadata found)")
    
    # Handle FSDP/DDP wrapped models by removing prefixes
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove FSDP/DDP prefixes
        clean_key = key
        if key.startswith('_fsdp_wrapped_module.'):
            clean_key = key.replace('_fsdp_wrapped_module.', '')
        elif key.startswith('module.'):
            clean_key = key.replace('module.', '')
        cleaned_state_dict[clean_key] = value
    
    # Extract model configuration from the state dict
    # This assumes standard GPT2Config values, but extracts what we can
    embed_weight = None
    pos_embed_weight = None
    n_layer = 0
    
    for key in cleaned_state_dict:
        if key == 'embed.weight':
            embed_weight = cleaned_state_dict[key]
        elif key == 'pos_embed.weight':
            pos_embed_weight = cleaned_state_dict[key]
        elif key.startswith('blocks.') and '.mha.c_attn.weight' in key:
            layer_num = int(key.split('.')[1])
            n_layer = max(n_layer, layer_num + 1)
    
    if embed_weight is None:
        raise ValueError("Could not find embed.weight in checkpoint")
    
    # Extract model dimensions
    n_vocab, n_embd = embed_weight.shape
    n_ctx = pos_embed_weight.shape[0] if pos_embed_weight is not None else CustomGPT2Config.n_ctx
    
    # Get n_head from attention weights
    first_attn_key = f'blocks.0.mha.c_attn.weight'
    if first_attn_key in cleaned_state_dict:
        attn_weight = cleaned_state_dict[first_attn_key]
        # c_attn has shape (n_embd, 3 * n_embd), and 3 * n_embd = 3 * n_head * head_dim
        # where head_dim = n_embd / n_head, so we can use the default
        n_head = CustomGPT2Config.n_head
    else:
        n_head = CustomGPT2Config.n_head
    
    print(f"Detected model config: n_vocab={n_vocab}, n_ctx={n_ctx}, n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
    
    # Create HuggingFace GPT2 config
    hf_config = GPT2Config(
        vocab_size=n_vocab,
        n_positions=n_ctx,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        use_cache=True,
        bos_token_id=50256,  # GPT2 tokenizer default
        eos_token_id=50256,
    )
    
    # Create HuggingFace model
    hf_model = GPT2LMHeadModel(hf_config)
    if out_dtype == "bfloat16":
        hf_model = hf_model.to(torch.bfloat16)
    
    # Convert state dict from our format to HuggingFace format
    hf_state_dict = {}
    
    # Token embeddings (tied with lm_head)
    hf_state_dict['transformer.wte.weight'] = cleaned_state_dict['embed.weight']
    hf_state_dict['lm_head.weight'] = cleaned_state_dict['embed.weight']  # Weight tying
    
    # Position embeddings
    if 'pos_embed.weight' in cleaned_state_dict:
        hf_state_dict['transformer.wpe.weight'] = cleaned_state_dict['pos_embed.weight']
    
    # Final layer norm
    if 'ln.weight' in cleaned_state_dict:
        hf_state_dict['transformer.ln_f.weight'] = cleaned_state_dict['ln.weight']
    if 'ln.bias' in cleaned_state_dict:
        hf_state_dict['transformer.ln_f.bias'] = cleaned_state_dict['ln.bias']
    
    # Transformer blocks
    for i in range(n_layer):
        block_prefix = f'blocks.{i}'
        hf_block_prefix = f'transformer.h.{i}'
        
        # Layer norm 1 (before attention)
        if f'{block_prefix}.norm1.weight' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.ln_1.weight'] = cleaned_state_dict[f'{block_prefix}.norm1.weight']
        if f'{block_prefix}.norm1.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.ln_1.bias'] = cleaned_state_dict[f'{block_prefix}.norm1.bias']
        
        # Attention weights (need to transpose)
        if f'{block_prefix}.mha.c_attn.weight' in cleaned_state_dict:
            # Transpose from [3*n_embd, n_embd] to [n_embd, 3*n_embd] for HF
            hf_state_dict[f'{hf_block_prefix}.attn.c_attn.weight'] = cleaned_state_dict[f'{block_prefix}.mha.c_attn.weight'].t()
        if f'{block_prefix}.mha.c_attn.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.attn.c_attn.bias'] = cleaned_state_dict[f'{block_prefix}.mha.c_attn.bias']
        
        # Attention projection (need to transpose)
        if f'{block_prefix}.mha.c_proj.weight' in cleaned_state_dict:
            # Transpose from [n_embd, n_embd] to [n_embd, n_embd] - this one might not need transpose, let's check
            hf_state_dict[f'{hf_block_prefix}.attn.c_proj.weight'] = cleaned_state_dict[f'{block_prefix}.mha.c_proj.weight'].t()
        if f'{block_prefix}.mha.c_proj.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.attn.c_proj.bias'] = cleaned_state_dict[f'{block_prefix}.mha.c_proj.bias']
        
        # Layer norm 2 (before MLP)
        if f'{block_prefix}.norm2.weight' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.ln_2.weight'] = cleaned_state_dict[f'{block_prefix}.norm2.weight']
        if f'{block_prefix}.norm2.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.ln_2.bias'] = cleaned_state_dict[f'{block_prefix}.norm2.bias']
        
        # MLP weights (need to transpose)
        if f'{block_prefix}.ff_up.weight' in cleaned_state_dict:
            # Transpose from [4*n_embd, n_embd] to [n_embd, 4*n_embd] for HF
            hf_state_dict[f'{hf_block_prefix}.mlp.c_fc.weight'] = cleaned_state_dict[f'{block_prefix}.ff_up.weight'].t()
        if f'{block_prefix}.ff_up.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.mlp.c_fc.bias'] = cleaned_state_dict[f'{block_prefix}.ff_up.bias']
        
        if f'{block_prefix}.ff_down.weight' in cleaned_state_dict:
            # Transpose from [n_embd, 4*n_embd] to [4*n_embd, n_embd] for HF
            hf_state_dict[f'{hf_block_prefix}.mlp.c_proj.weight'] = cleaned_state_dict[f'{block_prefix}.ff_down.weight'].t()
        if f'{block_prefix}.ff_down.bias' in cleaned_state_dict:
            hf_state_dict[f'{hf_block_prefix}.mlp.c_proj.bias'] = cleaned_state_dict[f'{block_prefix}.ff_down.bias']
    
    # Load the converted state dict
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in conversion: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in conversion: {unexpected_keys}")
    
    # Save the model
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, max_shard_size="5GB", safe_serialization=True)
    
    # Save tokenizer (use GPT2 tokenizer as default)
    print("Saving tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub if requested
    if push_to_hub and repo_name:
        print(f"Uploading {repo_name} to Hugging Face")
        hf_model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        print(f"Model uploaded to: https://huggingface.co/{repo_name}")
    
    print("Conversion complete!")
    return hf_model, tokenizer

def test_model(output_dir, prompt="The quick brown fox"):
    """Test the converted model with a simple generation"""
    print(f"Testing the converted model in {output_dir}...")
    print('-' * 80)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForCausalLM.from_pretrained(
            output_dir, 
            torch_dtype=torch.bfloat16, 
            device_map='auto' if torch.cuda.is_available() else 'cpu'
        )
        model.eval()
        
        # Generate text
        device = next(model.parameters()).device
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print(f"Prompt: {prompt}")
        print("Generated text:")
        
        with torch.no_grad():
            output = model.generate(
                tokens, 
                max_new_tokens=64, 
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Result: {generated_text}")
        print('-' * 80)
        
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure transformers is installed: pip install transformers")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to Hugging Face format')
    parser.add_argument("--input", "-i", help="Path to the PyTorch checkpoint (.pt file)", type=str, required=True)
    parser.add_argument("--output", "-o", help="Output directory for the Hugging Face model", type=str, required=True)
    parser.add_argument("--dtype", "-d", help="Output dtype (float32 or bfloat16)", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--push", action="store_true", help="Push the model to Hugging Face Hub")
    parser.add_argument("--repo_name", "-r", help="Repository name for Hugging Face Hub (required if --push is used)", type=str)
    parser.add_argument("--test", action="store_true", help="Test the converted model with text generation", default=True)
    parser.add_argument("--no_test", action="store_true", help="Skip testing the converted model")
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    if args.push and not args.repo_name:
        print("Error: --repo_name is required when --push is specified")
        sys.exit(1)
    
    # Convert the model
    try:
        convert_checkpoint_to_hf(
            args.input, 
            args.output, 
            push_to_hub=args.push, 
            repo_name=args.repo_name,
            out_dtype=args.dtype
        )
        
        # Test the model
        if args.test and not args.no_test:
            test_model(args.output)
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
