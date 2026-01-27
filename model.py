"""
GPT Model for Shakespeare Text Generation

This module implements a simplified GPT (Generative Pre-trained Transformer) model.
GPT is a decoder-only transformer that predicts the next token in a sequence.

Architecture Overview:
1. Token Embedding - Converts characters to vectors
2. Position Embedding - Adds position information
3. Transformer Blocks - Self-attention + Feed-forward layers
4. Output Layer - Predicts next character

This implementation uses PyTorch's built-in MultiheadAttention for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block

    Each block contains:
    1. Multi-Head Self-Attention (with causal mask)
    2. Feed-Forward Network (MLP)

    Both sublayers use:
    - Layer Normalization (applied before each sublayer - "Pre-LN")
    - Residual Connections (x + sublayer(x))
    """

    # 1. Initialize the class with hyperparameters
    def __init__(self,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        # 2. Create the first Layer Normalization
        self.ln1 = nn.LayerNorm(embedding_dim)

        # 3. Create Multi-Head Self-Attention using PyTorch's built-in module
        # This handles Q, K, V projections and attention computation internally
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, sequence, embedding)
        )

        # 4. Create the second Layer Normalization
        self.ln2 = nn.LayerNorm(embedding_dim)

        # 5. Create the Feed-Forward Network (MLP)
        # Expands to 4x, applies GELU, then projects back
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    # 6. Create the forward method
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            causal_mask: Attention mask to prevent looking at future tokens

        Returns:
            Output tensor of same shape as input
        """
        # 7. Self-Attention with residual connection
        # Pre-LN: normalize first, then apply attention
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=False  # We provide our own mask
        )
        x = x + attn_output  # Residual connection

        # 8. Feed-Forward with residual connection
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer)

    A decoder-only transformer for character-level text generation.
    Given a sequence of characters, it predicts the next character.

    Architecture:
    - Token Embedding: Maps character IDs to vectors
    - Position Embedding: Adds position information
    - N x Transformer Blocks: Process the sequence
    - Output Projection: Predicts next character probabilities
    """

    # 1. Initialize the class with hyperparameters
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # 2. Store block_size for generation
        self.block_size = block_size

        # 3. Create Token Embedding layer
        # Maps each character ID to a vector of size embedding_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # 4. Create Position Embedding layer
        # Each position (0, 1, 2, ..., block_size-1) gets its own vector
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

        # 5. Create Embedding Dropout
        self.dropout = nn.Dropout(dropout)

        # 6. Create stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 7. Create Final Layer Normalization
        self.ln_final = nn.LayerNorm(embedding_dim)

        # 8. Create Output Projection layer
        # Maps from embedding_dim to vocab_size (one score per character)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        # 9. Create causal mask (lower triangular) to prevent attending to future
        # True values are masked out (not allowed to attend)
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        # 10. Initialize weights
        self.apply(self._init_weights)

        # 11. Print model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model created with {total_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 12. Create the forward method
    def forward(self,
                input_ids: torch.Tensor,
                targets: torch.Tensor = None) -> tuple:
        """
        Forward pass of the GPT model.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            targets: Target token IDs for loss calculation (optional)

        Returns:
            logits: Predicted scores, shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 13. Get Token Embeddings
        token_emb = self.token_embedding(input_ids)

        # 14. Get Position Embeddings
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        # 15. Combine Token and Position Embeddings
        x = self.dropout(token_emb + pos_emb)

        # 16. Get the causal mask for current sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # 17. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        # 18. Apply Final Layer Normalization
        x = self.ln_final(x)

        # 19. Project to vocabulary size
        logits = self.output_proj(x)

        # 20. Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    # 21. Create the generate method for text generation
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int = None) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            input_ids: Starting tokens, shape (batch_size, sequence_length)
            max_new_tokens: Number of new tokens to generate
            temperature: Higher = more random, Lower = more deterministic
            top_k: Only sample from the top k most likely tokens

        Returns:
            Generated tokens, shape (batch_size, sequence_length + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 22. Crop to last block_size tokens if sequence is too long
            idx_cond = input_ids if input_ids.size(1) <= self.block_size else input_ids[:, -self.block_size:]

            # 23. Get predictions
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature  # Get last position and scale

            # 24. Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 25. Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 26. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Test the model
if __name__ == "__main__":
    # 1. Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create model
    model = GPT(
        vocab_size=65,
        embedding_dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256
    ).to(device)

    # 3. Create dummy input
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, 65, (batch_size, seq_len)).to(device)

    # 4. Test forward pass
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # 5. Test generation
    generated = model.generate(dummy_input[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
