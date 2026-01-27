"""
Text Generation Script for Shakespeare GPT

This module demonstrates how to use the trained model to generate Shakespeare-like text.

Key Concepts:
1. Loading a trained model from checkpoint
2. Autoregressive generation (one character at a time)
3. Temperature and top-k sampling for controlling randomness
"""

import torch
from model import GPT
from dataset import download_shakespeare, CharacterTokenizer, DATA_PATH


def convert_old_checkpoint(old_state_dict: dict, config: dict) -> dict:
    """
    Convert state_dict from old custom attention model to new nn.MultiheadAttention model.

    The old model uses separate weights for each attention head:
        blocks.X.attention.heads.Y.query/key/value.weight

    The new model uses nn.MultiheadAttention which combines all heads:
        blocks.X.attention.in_proj_weight (Q, K, V concatenated)
        blocks.X.attention.out_proj.weight
    """
    new_state_dict = {}
    num_heads = config.get('num_heads', 6)
    num_layers = config.get('num_layers', 6)
    embedding_dim = config.get('embedding_dim') or config.get('embed_dim', 384)
    block_size = config.get('block_size', 256)

    for key, value in old_state_dict.items():
        # Skip attention head weights - we'll handle them separately
        if '.attention.heads.' in key:
            continue
        if '.attention.output_proj.' in key:
            continue
        # Skip mask buffers from old model
        if '.mask' in key:
            continue

        new_key = key

        # Rename feed_forward.net -> mlp
        new_key = new_key.replace('.feed_forward.net.', '.mlp.')

        new_state_dict[new_key] = value

    # Now handle attention weights for each layer
    for layer_idx in range(num_layers):
        # Collect Q, K, V weights from all heads
        q_weights = []
        k_weights = []
        v_weights = []

        for head_idx in range(num_heads):
            q_key = f'blocks.{layer_idx}.attention.heads.{head_idx}.query.weight'
            k_key = f'blocks.{layer_idx}.attention.heads.{head_idx}.key.weight'
            v_key = f'blocks.{layer_idx}.attention.heads.{head_idx}.value.weight'

            if q_key in old_state_dict:
                q_weights.append(old_state_dict[q_key])
                k_weights.append(old_state_dict[k_key])
                v_weights.append(old_state_dict[v_key])

        if q_weights:
            # Concatenate head weights: each is (head_size, embed_dim)
            # Result for each of Q, K, V: (num_heads * head_size, embed_dim) = (embed_dim, embed_dim)
            q_combined = torch.cat(q_weights, dim=0)  # (embed_dim, embed_dim)
            k_combined = torch.cat(k_weights, dim=0)  # (embed_dim, embed_dim)
            v_combined = torch.cat(v_weights, dim=0)  # (embed_dim, embed_dim)

            # nn.MultiheadAttention expects in_proj_weight as (3*embed_dim, embed_dim)
            # Order: Q, K, V stacked
            in_proj_weight = torch.cat([q_combined, k_combined, v_combined], dim=0)
            new_state_dict[f'blocks.{layer_idx}.attention.in_proj_weight'] = in_proj_weight

            # Add zero bias for in_proj (old model didn't have bias)
            in_proj_bias = torch.zeros(3 * embedding_dim)
            new_state_dict[f'blocks.{layer_idx}.attention.in_proj_bias'] = in_proj_bias

            # Handle output projection
            out_proj_w_key = f'blocks.{layer_idx}.attention.output_proj.weight'
            out_proj_b_key = f'blocks.{layer_idx}.attention.output_proj.bias'

            if out_proj_w_key in old_state_dict:
                new_state_dict[f'blocks.{layer_idx}.attention.out_proj.weight'] = old_state_dict[out_proj_w_key]
            if out_proj_b_key in old_state_dict:
                new_state_dict[f'blocks.{layer_idx}.attention.out_proj.bias'] = old_state_dict[out_proj_b_key]

    # Add causal mask (this is a buffer, not a learned parameter)
    # Upper triangular matrix with True values (positions to mask)
    causal_mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
    new_state_dict['causal_mask'] = causal_mask

    return new_state_dict


def load_model(checkpoint_path: str = "checkpoints/best.pt",
               device: str = "mps") -> tuple:
    """
    Load a trained GPT model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device to load the model on ('mps', 'cuda', or 'cpu')

    Returns:
        model: The loaded GPT model
        tokenizer: The character tokenizer
    """
    # 1. Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Get the configuration from checkpoint
    config = checkpoint['config']

    # 3. Download Shakespeare text and create tokenizer
    download_shakespeare()
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = CharacterTokenizer(text)

    # 4. Create the model with saved configuration
    # Handle both old key names (embed_dim) and new key names (embedding_dim)
    embedding_dim = config.get('embedding_dim') or config.get('embed_dim')
    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=embedding_dim,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0  # No dropout during inference
    )

    # 5. Load the saved weights
    state_dict = checkpoint['model_state_dict']

    # Check if this is an old checkpoint (uses custom multi-head attention with separate heads)
    if any('.attention.heads.' in k for k in state_dict.keys()):
        print("  Converting old checkpoint format (custom attention) to new format (nn.MultiheadAttention)...")
        state_dict = convert_old_checkpoint(state_dict, config)

    model.load_state_dict(state_dict)

    # 6. Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # 7. Print info
    # Handle both old key names (iter_num) and new key names (iteration)
    iteration = checkpoint.get('iteration') or checkpoint.get('iter_num', 'unknown')
    print(f"Model loaded successfully!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {iteration}")

    return model, tokenizer


@torch.no_grad()
def generate(model: torch.nn.Module,
             tokenizer,
             prompt: str = "ROMEO:",
             max_tokens: int = 500,
             temperature: float = 0.8,
             top_k: int = 40,
             device: str = "mps") -> str:
    """
    Generate text given a prompt.

    Args:
        model: The trained GPT model
        tokenizer: The character tokenizer
        prompt: Starting text for generation
        max_tokens: Maximum number of characters to generate
        temperature: Controls randomness (higher = more random)
        top_k: Only sample from top k most likely characters
        device: Device to run on

    Returns:
        Generated text as a string
    """
    # 1. Encode the prompt to token IDs
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # Add batch dimension: (1, seq_len)

    # 2. Generate new tokens
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # 3. Decode the output to text
    generated_text = tokenizer.decode(output_ids[0])

    return generated_text


def interactive_mode(checkpoint_path: str = "checkpoints/best.pt"):
    """
    Interactive text generation mode.

    Allows users to enter prompts and generate Shakespeare-like text.
    """
    # 1. Set up device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 2. Load model
    model, tokenizer = load_model(checkpoint_path, device)

    # 3. Print instructions
    print("\n" + "=" * 60)
    print("Shakespeare GPT - Interactive Generation")
    print("=" * 60)
    print("\nEnter a prompt to generate Shakespeare-like text.")
    print("Try character names like 'ROMEO:', 'JULIET:', 'HAMLET:'")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'temp X' - Set temperature (e.g., 'temp 0.5')")
    print("  'topk X' - Set top_k (e.g., 'topk 20')")
    print("\nDefault: temperature=0.8, top_k=40")
    print("-" * 60)

    # 4. Set default parameters
    temperature = 0.8
    top_k = 40

    # 5. Main interaction loop
    while True:
        try:
            # Get user input
            prompt = input("\nPrompt: ")

            # Skip empty input
            if not prompt:
                continue

            # Check for exit commands
            if prompt.lower() in ['quit', 'exit']:
                print("Farewell!")
                break

            # Check for temperature command
            if prompt.lower().startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Invalid temperature value. Example: 'temp 0.5'")
                continue

            # Check for top_k command
            if prompt.lower().startswith('topk '):
                try:
                    top_k = int(prompt.split()[1])
                    print(f"Top-k set to {top_k}")
                except:
                    print("Invalid top_k value. Example: 'topk 20'")
                continue

            # Generate text
            print("\nGenerating...")
            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=500,
                temperature=temperature,
                top_k=top_k,
                device=device
            )

            # Print the result
            print("\n" + "-" * 60)
            print(generated_text)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break


def demo_mode(checkpoint_path: str = "checkpoints/best.pt"):
    """
    Demo mode: Generate text with various prompts and settings.
    """
    # 1. Set up device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 2. Load model
    model, tokenizer = load_model(checkpoint_path, device)

    # 3. Define demo prompts
    prompts = [
        "ROMEO:\nO, ",
        "JULIET:\nMy love, ",
        "HAMLET:\nTo be, or ",
        "First Citizen:\nWe are ",
        "KING:\nNow hear me, "
    ]

    # 4. Print header
    print("\n" + "=" * 60)
    print("Shakespeare GPT - Demo Generation")
    print("=" * 60)

    # 5. Generate with each prompt
    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {repr(prompt)}")
        print("=" * 60)

        # Try different temperatures
        for temperature in [0.5, 1.0]:
            print(f"\n[Temperature: {temperature}]")
            print("-" * 40)

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=300,
                temperature=temperature,
                top_k=40,
                device=device
            )

            print(generated_text)


if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate Shakespeare-like text")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "demo"],
        default="interactive",
        help="Generation mode: 'interactive' or 'demo'"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Generate with a single prompt and exit"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of characters to generate"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate mode
    if args.prompt:
        # Single generation mode
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, tokenizer = load_model(args.checkpoint, device)

        text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print(text)

    elif args.mode == "demo":
        demo_mode(args.checkpoint)

    else:
        interactive_mode(args.checkpoint)
