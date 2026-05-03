"""
Transformers Basics for Machine Learning
Practical Python examples implementing Transformer architecture from scratch
Based on "Attention Is All You Need"
"""

import numpy as np


def section_divider(title):
    """Print a formatted section divider"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_attention_mechanism():
    """Demonstrate basic attention mechanism"""
    section_divider("1. ATTENTION MECHANISM")
    
    def softmax(x):
        """Compute softmax values for numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def attention(Q, K, V):
        """
        Compute attention scores
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        Returns: output, attention_weights
        """
        d_k = K.shape[-1]
        
        # Step 1: Compute attention scores (Q × K^T)
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        print(f"Attention scores (Q × K^T / √d_k):")
        print(scores)
        print(f"Shape: {scores.shape}\n")
        
        # Step 2: Apply softmax to get attention weights
        attention_weights = softmax(scores)
        print(f"Attention weights (after softmax):")
        print(attention_weights)
        print(f"Note: Each row sums to 1.0")
        print(f"Row sums: {attention_weights.sum(axis=-1)}\n")
        
        # Step 3: Multiply weights by values
        output = np.dot(attention_weights, V)
        print(f"Attention output (weights × V):")
        print(output)
        print(f"Shape: {output.shape}")
        
        return output, attention_weights
    
    # Example with 3 tokens, dimension 4
    print("Example: Sentence with 3 tokens")
    print("Dimension d_k = 4\n")
    
    seq_len = 3
    d_k = 4
    
    # Create Q, K, V matrices (in practice, these come from input embeddings)
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    print(f"Query (Q) - 'What am I looking for?':")
    print(f"Shape: {Q.shape}\n")
    
    print(f"Key (K) - 'What do I have?':")
    print(f"Shape: {K.shape}\n")
    
    print(f"Value (V) - 'What information do I return?':")
    print(f"Shape: {V.shape}\n")
    
    output, weights = attention(Q, K, V)
    
    print("\nIntuition:")
    print("- Each token 'queries' all other tokens (including itself)")
    print("- Attention weights show how much to focus on each token")
    print("- Output is weighted combination of all values")


def demo_self_attention():
    """Demonstrate self-attention where Q, K, V come from same input"""
    section_divider("2. SELF-ATTENTION")
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def attention(Q, K, V):
        d_k = K.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output = np.dot(attention_weights, V)
        return output, attention_weights
    
    def self_attention(X, W_q, W_k, W_v):
        """
        Self-attention: Q, K, V all derived from same input X
        X: Input embeddings (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices for projections
        """
        # Project input to Q, K, V
        Q = np.dot(X, W_q)  # (seq_len, d_k)
        K = np.dot(X, W_k)  # (seq_len, d_k)
        V = np.dot(X, W_v)  # (seq_len, d_v)
        
        print("Input X projected to:")
        print(f"  Q (Query) shape: {Q.shape}")
        print(f"  K (Key) shape: {K.shape}")
        print(f"  V (Value) shape: {V.shape}\n")
        
        # Compute attention
        output, weights = attention(Q, K, V)
        return output, weights
    
    print("Self-Attention: Each token attends to all tokens in the sequence\n")
    print("Example: 'The cat sat on the mat'")
    print("When processing 'sat', it can attend to 'cat', 'the', etc.\n")
    
    # Simulate 4 tokens with embedding dimension 8
    seq_len = 4
    d_model = 8  # Embedding dimension
    d_k = 4      # Query/Key dimension (typically d_model // num_heads)
    
    np.random.seed(42)
    
    # Input embeddings (e.g., word embeddings)
    X = np.random.randn(seq_len, d_model)
    print(f"Input embeddings X:")
    print(f"Shape: {X.shape} (4 tokens, 8-dim embeddings)\n")
    
    # Weight matrices to project to Q, K, V
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_k) * 0.1
    
    # Compute self-attention
    output, weights = self_attention(X, W_q, W_k, W_v)
    
    print("Attention weights matrix:")
    print(weights)
    print("\nInterpretation:")
    print("- Row i, Column j = how much token i attends to token j")
    print("- Example: weights[0, 1] = attention from token 0 to token 1")
    print(f"\nOutput shape: {output.shape}")
    print("Each token now contains information from all tokens it attended to!")


def demo_multi_head_attention():
    """Demonstrate multi-head attention"""
    section_divider("3. MULTI-HEAD ATTENTION")
    
    class MultiHeadAttention:
        def __init__(self, d_model, num_heads):
            """
            d_model: Embedding dimension (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
            """
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads  # Dimension per head
            
            # Initialize weight matrices
            np.random.seed(42)
            self.W_q = np.random.randn(d_model, d_model) * 0.01
            self.W_k = np.random.randn(d_model, d_model) * 0.01
            self.W_v = np.random.randn(d_model, d_model) * 0.01
            self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        def split_heads(self, x, batch_size):
            """
            Split the last dimension into (num_heads, d_k)
            x: (batch_size, seq_len, d_model)
            Returns: (batch_size, num_heads, seq_len, d_k)
            """
            seq_len = x.shape[1]
            x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
            return x.transpose(0, 2, 1, 3)
        
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / exp_x.sum(axis=-1, keepdims=True)
        
        def attention(self, Q, K, V):
            """Compute scaled dot-product attention"""
            d_k = K.shape[-1]
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
            attention_weights = self.softmax(scores)
            output = np.matmul(attention_weights, V)
            return output, attention_weights
        
        def forward(self, X):
            """
            X: Input (batch_size, seq_len, d_model)
            """
            batch_size, seq_len, _ = X.shape
            
            # Linear projections
            Q = np.dot(X, self.W_q)  # (batch, seq_len, d_model)
            K = np.dot(X, self.W_k)
            V = np.dot(X, self.W_v)
            
            print(f"Step 1: Project input to Q, K, V")
            print(f"  Q shape: {Q.shape}")
            print(f"  K shape: {K.shape}")
            print(f"  V shape: {V.shape}\n")
            
            # Split into multiple heads
            Q = self.split_heads(Q, batch_size)  # (batch, heads, seq_len, d_k)
            K = self.split_heads(K, batch_size)
            V = self.split_heads(V, batch_size)
            
            print(f"Step 2: Split into {self.num_heads} heads")
            print(f"  Q shape: {Q.shape} (batch, heads, seq_len, d_k)")
            print(f"  Each head has dimension: {self.d_k}\n")
            
            # Compute attention for all heads in parallel
            output, attention_weights = self.attention(Q, K, V)
            
            print(f"Step 3: Compute attention for each head")
            print(f"  Output shape: {output.shape}\n")
            print(f"  Attention weights shape: {attention_weights.shape}")
            print(f"  (batch, heads, seq_len, seq_len)\n")
            
            # Concatenate heads
            output = output.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_k)
            output = output.reshape(batch_size, seq_len, self.d_model)
            
            print(f"Step 4: Concatenate heads")
            print(f"  Concatenated shape: {output.shape}\n")
            
            # Final linear projection
            output = np.dot(output, self.W_o)
            
            print(f"Step 5: Final linear projection")
            print(f"  Final output shape: {output.shape}")
            
            return output, attention_weights
    
    print("Multi-Head Attention: Run attention multiple times in parallel\n")
    print("Why multiple heads?")
    print("  - Head 1 might focus on syntax (subject-verb relationships)")
    print("  - Head 2 might focus on semantics (word meanings)")
    print("  - Head 3 might focus on long-range dependencies")
    print("  - Each head learns different patterns!\n")
    
    # Example
    d_model = 512  # Embedding dimension
    num_heads = 8
    batch_size = 1
    seq_len = 10
    
    print(f"Configuration:")
    print(f"  d_model (embedding dim): {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (dim per head): {d_model // num_heads}")
    print(f"  seq_len: {seq_len}\n")
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Random input
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"Input X shape: {X.shape} (batch, seq_len, d_model)\n")
    print("-" * 70)
    
    # Forward pass
    output, attention_weights = mha.forward(X)
    
    print("-" * 70)
    print(f"\n✓ Multi-head attention complete!")
    print(f"  Output shape: {output.shape}")
    print(f"  Contains information from all {num_heads} attention heads")


def demo_positional_encoding():
    """Demonstrate positional encoding"""
    section_divider("4. POSITIONAL ENCODING")
    
    def positional_encoding(seq_len, d_model):
        """
        Generate sinusoidal positional encodings
        seq_len: Length of sequence
        d_model: Embedding dimension
        
        Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        # Initialize position and dimension indices
        pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]     # (1, d_model)
        
        # Calculate angle rates
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle = pos * angle_rates
        
        # Apply sin to even indices, cos to odd indices
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angle[:, 0::2])  # Even indices
        pos_encoding[:, 1::2] = np.cos(angle[:, 1::2])  # Odd indices
        
        return pos_encoding
    
    print("Why Positional Encoding?")
    print("  - Transformers process sequences in parallel")
    print("  - They have no inherent notion of token order")
    print("  - 'Dog bites man' vs 'Man bites dog' - order matters!")
    print("  - Positional encoding adds position information\n")
    
    print("Approach: Sinusoidal functions")
    print("  - Use sine for even dimensions")
    print("  - Use cosine for odd dimensions")
    print("  - Different frequencies for different dimensions\n")
    
    # Example
    seq_len = 10
    d_model = 8
    
    print(f"Generating positional encodings:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {d_model}\n")
    
    pos_enc = positional_encoding(seq_len, d_model)
    
    print(f"Positional encoding matrix:")
    print(f"Shape: {pos_enc.shape} (seq_len, d_model)")
    print(pos_enc)
    
    print("\n" + "-" * 70)
    print("How to use:")
    print("  1. Get word embeddings: X = Embedding(tokens)")
    print("  2. Add positional encoding: X = X + PositionalEncoding")
    print("  3. Now each token has both content AND position information!")
    
    # Demonstrate with example embeddings
    print("\n" + "-" * 70)
    print("Example: Adding positional encoding to embeddings\n")
    
    np.random.seed(42)
    embeddings = np.random.randn(seq_len, d_model)
    
    print(f"Word embeddings (random):")
    print(f"Shape: {embeddings.shape}")
    print(embeddings[:3])  # Show first 3
    
    print(f"\nPositional encodings:")
    print(pos_enc[:3])  # Show first 3
    
    # Add them together
    embeddings_with_pos = embeddings + pos_enc
    
    print(f"\nFinal embeddings (word + position):")
    print(embeddings_with_pos[:3])  # Show first 3
    print(f"\n✓ Each token now has content + position information!")


def demo_feedforward_network():
    """Demonstrate position-wise feed-forward network"""
    section_divider("5. FEED-FORWARD NETWORK")
    
    class FeedForward:
        def __init__(self, d_model, d_ff):
            """
            Position-wise Feed-Forward Network
            d_model: Input/output dimension (e.g., 512)
            d_ff: Hidden layer dimension (e.g., 2048)
            
            Formula: FFN(x) = ReLU(x·W1 + b1)·W2 + b2
            """
            np.random.seed(42)
            self.W1 = np.random.randn(d_model, d_ff) * 0.01
            self.b1 = np.zeros(d_ff)
            self.W2 = np.random.randn(d_ff, d_model) * 0.01
            self.b2 = np.zeros(d_model)
            
            self.d_model = d_model
            self.d_ff = d_ff
        
        def relu(self, x):
            """ReLU activation: max(0, x)"""
            return np.maximum(0, x)
        
        def forward(self, x):
            """
            x: Input (batch_size, seq_len, d_model)
            """
            print(f"Input shape: {x.shape}")
            
            # First linear transformation + ReLU
            hidden = np.dot(x, self.W1) + self.b1
            print(f"After W1 (before ReLU): {hidden.shape}")
            print(f"  Expanded from {self.d_model} to {self.d_ff} dimensions")
            
            activated = self.relu(hidden)
            print(f"After ReLU activation: {activated.shape}")
            print(f"  Negative values zeroed out")
            
            # Second linear transformation
            output = np.dot(activated, self.W2) + self.b2
            print(f"After W2 (final): {output.shape}")
            print(f"  Compressed back to {self.d_model} dimensions")
            
            return output
    
    print("Feed-Forward Network (FFN):")
    print("  - Applied to each position independently")
    print("  - Two linear transformations with ReLU in between")
    print("  - Expands then compresses dimensionality")
    print("  - Same network applied to every position\n")
    
    print("Architecture:")
    print("  Input (d_model) → Linear → ReLU → Linear → Output (d_model)")
    print("                     ↓                  ↑")
    print("                   d_ff (larger)       \n")
    
    # Example
    d_model = 512
    d_ff = 2048  # Typically 4x d_model
    batch_size = 1
    seq_len = 10
    
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff} (4x d_model)")
    print(f"  sequence length: {seq_len}\n")
    
    ff = FeedForward(d_model, d_ff)
    
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print("-" * 70)
    output = ff.forward(x)
    print("-" * 70)
    
    print(f"\n✓ Feed-forward complete!")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Shape preserved: {x.shape == output.shape}")


def demo_layer_normalization():
    """Demonstrate layer normalization"""
    section_divider("6. LAYER NORMALIZATION")
    
    class LayerNorm:
        def __init__(self, d_model, eps=1e-6):
            """
            Layer Normalization
            d_model: Dimension to normalize over
            eps: Small constant for numerical stability
            
            Formula: LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
            """
            self.gamma = np.ones(d_model)   # Scale parameter
            self.beta = np.zeros(d_model)   # Shift parameter
            self.eps = eps
            self.d_model = d_model
        
        def forward(self, x):
            """
            x: Input (batch_size, seq_len, d_model)
            Normalize across the last dimension (features)
            """
            print(f"Input shape: {x.shape}\n")
            
            # Compute mean and variance across last dimension
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            
            print(f"Mean shape: {mean.shape}")
            print(f"Sample means: {mean[0, :3, 0]}")
            
            print(f"\nVariance shape: {var.shape}")
            print(f"Sample variances: {var[0, :3, 0]}")
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            print(f"\nNormalized (x - μ) / √(σ² + ε):")
            print(f"  Mean ≈ 0: {x_norm[0, 0].mean():.6f}")
            print(f"  Std ≈ 1: {x_norm[0, 0].std():.6f}")
            
            # Scale and shift
            output = self.gamma * x_norm + self.beta
            
            print(f"\nAfter scale (γ) and shift (β):")
            print(f"  Output shape: {output.shape}")
            
            return output
    
    print("Layer Normalization:")
    print("  - Normalizes inputs across features (last dimension)")
    print("  - Makes training more stable")
    print("  - Reduces internal covariate shift")
    print("  - Different from Batch Normalization (normalizes across batch)\n")
    
    print("Steps:")
    print("  1. Compute mean μ and variance σ² across features")
    print("  2. Normalize: (x - μ) / √(σ² + ε)")
    print("  3. Scale and shift: γ ⊙ x_norm + β")
    print("  4. γ and β are learnable parameters\n")
    
    # Example
    d_model = 512
    batch_size = 1
    seq_len = 5
    
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  sequence length: {seq_len}\n")
    
    ln = LayerNorm(d_model)
    
    np.random.seed(42)
    # Create input with some variation
    x = np.random.randn(batch_size, seq_len, d_model) * 2 + 5
    
    print("-" * 70)
    output = ln.forward(x)
    print("-" * 70)
    
    print(f"\n✓ Layer normalization complete!")
    print(f"  Each token normalized independently across its features")


def demo_encoder_layer():
    """Demonstrate complete encoder layer"""
    section_divider("7. TRANSFORMER ENCODER LAYER")
    
    print("Encoder Layer Components:")
    print("  1. Multi-Head Self-Attention")
    print("  2. Add & Norm (Residual connection + Layer Norm)")
    print("  3. Feed-Forward Network")
    print("  4. Add & Norm\n")
    
    print("Flow:")
    print("  Input → [Multi-Head Attention] → Add & Norm →")
    print("          [Feed-Forward Network] → Add & Norm → Output\n")
    
    print("Residual Connections (Add & Norm):")
    print("  - Add input to output: output = LayerNorm(input + sublayer(input))")
    print("  - Helps gradients flow during backpropagation")
    print("  - Allows training very deep networks\n")
    
    # Simplified demonstration
    d_model = 512
    seq_len = 10
    
    print(f"Simplified Example:")
    print(f"  d_model: {d_model}")
    print(f"  seq_len: {seq_len}\n")
    
    np.random.seed(42)
    x = np.random.randn(1, seq_len, d_model)
    
    print(f"Step 1: Input")
    print(f"  Shape: {x.shape}\n")
    
    # Simulate attention output
    attn_output = np.random.randn(1, seq_len, d_model) * 0.1
    print(f"Step 2: Multi-Head Attention")
    print(f"  Output shape: {attn_output.shape}\n")
    
    # Add & Norm 1
    x1 = x + attn_output  # Residual connection
    # Would apply LayerNorm here
    print(f"Step 3: Add & Norm (residual connection)")
    print(f"  x = x + attention(x)")
    print(f"  Shape: {x1.shape}\n")
    
    # Simulate FFN output
    ffn_output = np.random.randn(1, seq_len, d_model) * 0.1
    print(f"Step 4: Feed-Forward Network")
    print(f"  Output shape: {ffn_output.shape}\n")
    
    # Add & Norm 2
    x2 = x1 + ffn_output  # Residual connection
    print(f"Step 5: Add & Norm (residual connection)")
    print(f"  x = x + FFN(x)")
    print(f"  Final shape: {x2.shape}\n")
    
    print("✓ Encoder layer complete!")
    print("  Original shape preserved: ", x.shape == x2.shape)
    print("\nKey insight: Shape stays constant through the layer!")
    print("  This allows stacking multiple layers (typically 6-12)")


def demo_decoder_layer():
    """Demonstrate decoder layer differences"""
    section_divider("8. TRANSFORMER DECODER LAYER")
    
    print("Decoder Layer Components:")
    print("  1. Masked Multi-Head Self-Attention (can't see future)")
    print("  2. Add & Norm")
    print("  3. Multi-Head Cross-Attention (attend to encoder)")
    print("  4. Add & Norm")
    print("  5. Feed-Forward Network")
    print("  6. Add & Norm\n")
    
    print("Key Difference from Encoder:")
    print("  1. Masked Self-Attention:")
    print("     - During training, can't look at future tokens")
    print("     - Example: When predicting token 3, can only see tokens 0, 1, 2")
    print("     - Prevents 'cheating' during training\n")
    
    print("  2. Cross-Attention:")
    print("     - Q comes from decoder")
    print("     - K, V come from encoder output")
    print("     - Allows decoder to focus on relevant encoder outputs\n")
    
    # Demonstrate masking
    print("Masking Example:")
    print("  Original attention scores:\n")
    
    seq_len = 4
    scores = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.4, 1.0, 0.6, 0.1],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.4, 1.0]
    ])
    print(scores)
    
    print("\n  Look-ahead mask (1 = allowed, 0 = masked):\n")
    mask = np.tril(np.ones((seq_len, seq_len)))
    print(mask.astype(int))
    
    print("\n  After masking (set -inf before softmax):\n")
    masked_scores = scores.copy()
    masked_scores[mask == 0] = -np.inf
    print(masked_scores)
    
    print("\n  Interpretation:")
    print("    - Token 0 can only see token 0")
    print("    - Token 1 can see tokens 0, 1")
    print("    - Token 2 can see tokens 0, 1, 2")
    print("    - Token 3 can see all previous tokens\n")
    
    print("✓ This ensures autoregressive generation:")
    print("  Each token is predicted based only on previous tokens!")


def demo_complete_transformer():
    """Demonstrate the complete transformer architecture"""
    section_divider("9. COMPLETE TRANSFORMER ARCHITECTURE")
    
    print("Full Transformer Architecture:\n")
    
    print("┌─────────────────────────────────────────┐")
    print("│            INPUT SEQUENCE                │")
    print("└─────────────────────────────────────────┘")
    print("                    ↓")
    print("        ┌──────────────────────┐")
    print("        │   Input Embedding    │")
    print("        │  + Positional Enc.   │")
    print("        └──────────────────────┘")
    print("                    ↓")
    print("        ┌──────────────────────┐")
    print("        │   ENCODER STACK      │")
    print("        │  (6-12 layers)       │")
    print("        │  - Multi-Head Attn   │")
    print("        │  - Feed-Forward      │")
    print("        │  - Add & Norm        │")
    print("        └──────────────────────┘")
    print("                    ↓")
    print("           [Encoder Output]")
    print("                    ↓")
    print("        ┌──────────────────────┐")
    print("        │   DECODER STACK      │")
    print("        │  (6-12 layers)       │")
    print("        │  - Masked Self-Attn  │")
    print("        │  - Cross-Attention   │")
    print("        │  - Feed-Forward      │")
    print("        │  - Add & Norm        │")
    print("        └──────────────────────┘")
    print("                    ↓")
    print("        ┌──────────────────────┐")
    print("        │   Linear + Softmax   │")
    print("        └──────────────────────┘")
    print("                    ↓")
    print("┌─────────────────────────────────────────┐")
    print("│           OUTPUT SEQUENCE                │")
    print("└─────────────────────────────────────────┘\n")
    
    # Example parameters
    print("Typical Hyperparameters:\n")
    
    configs = [
        ("BERT-Base", 512, 8, 12, 2048, 110),
        ("GPT-2", 768, 12, 12, 3072, 117),
        ("Transformer-Base", 512, 8, 6, 2048, 65),
        ("Transformer-Big", 1024, 16, 6, 4096, 213),
    ]
    
    print(f"{'Model':<20} {'d_model':<10} {'heads':<8} {'layers':<8} {'d_ff':<10} {'params (M)'}")
    print("-" * 70)
    for name, d_model, heads, layers, d_ff, params in configs:
        print(f"{name:<20} {d_model:<10} {heads:<8} {layers:<8} {d_ff:<10} {params}")
    
    print("\n\nKey Design Choices:")
    print("  • d_model: Usually 512 or 1024")
    print("  • num_heads: Typically 8 or 16 (d_model must be divisible)")
    print("  • num_layers: 6-12 for base models, more for large models")
    print("  • d_ff: Usually 4× d_model")
    print("  • dropout: 0.1 for regularization")


def demo_transformer_variants():
    """Demonstrate different transformer variants"""
    section_divider("10. TRANSFORMER VARIANTS")
    
    print("Modern Transformer Variants:\n")
    
    print("1. ENCODER-DECODER (Original Transformer)")
    print("   Use case: Sequence-to-sequence tasks")
    print("   Examples: Machine translation, summarization")
    print("   Models: T5, BART, Original Transformer\n")
    
    print("2. ENCODER-ONLY")
    print("   Use case: Understanding tasks (classification, NER)")
    print("   Examples:")
    print("     • BERT - Masked language modeling")
    print("     • RoBERTa - Optimized BERT")
    print("     • DistilBERT - Smaller, faster BERT")
    print("   Key: Bidirectional context (can see full sentence)\n")
    
    print("3. DECODER-ONLY")
    print("   Use case: Text generation, autoregressive tasks")
    print("   Examples:")
    print("     • GPT (1, 2, 3, 4) - Generative pre-training")
    print("     • Claude - Conversational AI")
    print("     • LLaMA - Efficient large language model")
    print("     • PaLM - Pathways Language Model")
    print("   Key: Causal (left-to-right) attention only\n")
    
    print("-" * 70)
    print("\nComparison:\n")
    
    print(f"{'Type':<20} {'Attention':<20} {'Best For'}")
    print("-" * 70)
    print(f"{'Encoder-Decoder':<20} {'Full + Causal':<20} {'Translation, summarization'}")
    print(f"{'Encoder-Only':<20} {'Bidirectional':<20} {'Classification, NER'}")
    print(f"{'Decoder-Only':<20} {'Causal':<20} {'Text generation, chat'}")
    
    print("\n" + "-" * 70)
    print("\nWhy Decoder-Only Dominates Now?")
    print("  1. Simpler architecture (just stack decoder layers)")
    print("  2. Scales very well (GPT-3: 175B parameters)")
    print("  3. Can be fine-tuned for many tasks")
    print("  4. Excellent at few-shot learning")
    print("  5. Unified approach for generation and understanding")


def demo_practical_applications():
    """Demonstrate practical transformer applications"""
    section_divider("11. PRACTICAL APPLICATIONS")
    
    print("Real-World Transformer Applications:\n")
    
    applications = [
        ("Machine Translation", "English → French, etc.", "Encoder-Decoder"),
        ("Text Summarization", "Long article → Short summary", "Encoder-Decoder"),
        ("Question Answering", "Context + Question → Answer", "Encoder or Encoder-Decoder"),
        ("Sentiment Analysis", "Text → Positive/Negative", "Encoder"),
        ("Named Entity Recognition", "Find people, places, orgs", "Encoder"),
        ("Text Generation", "Prompt → Generated text", "Decoder"),
        ("Code Generation", "Description → Code", "Decoder"),
        ("Chatbots", "Conversation → Response", "Decoder"),
    ]
    
    print(f"{'Task':<25} {'Description':<30} {'Architecture'}")
    print("-" * 70)
    for task, desc, arch in applications:
        print(f"{task:<25} {desc:<30} {arch}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Machine Translation")
    print("=" * 70 + "\n")
    
    print("Input:  'Hello, how are you?'")
    print("        ↓")
    print("Step 1: Tokenize → [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]")
    print("        ↓")
    print("Step 2: Embed → Convert tokens to vectors (512-dim)")
    print("        ↓")
    print("Step 3: Add positional encoding")
    print("        ↓")
    print("Step 4: Pass through ENCODER (6 layers)")
    print("        - Each layer: Attention → FFN → Norm")
    print("        ↓")
    print("Step 5: Encoder output → Context vectors")
    print("        ↓")
    print("Step 6: Pass through DECODER (6 layers)")
    print("        - Start with <START> token")
    print("        - Generate one token at a time")
    print("        - Use encoder output via cross-attention")
    print("        ↓")
    print("Step 7: Output → 'Bonjour, comment allez-vous?'")
    
    print("\n" + "=" * 70)
    print("EXAMPLE: GPT Text Generation")
    print("=" * 70 + "\n")
    
    print("Prompt: 'Once upon a time'")
    print("        ↓")
    print("Step 1: Embed + Positional encoding")
    print("        ↓")
    print("Step 2: Pass through DECODER layers (causal attention)")
    print("        ↓")
    print("Step 3: Predict next token → 'there'")
    print("        ↓")
    print("Step 4: Add 'there' to sequence, repeat")
    print("        ↓")
    print("Output: 'Once upon a time there was a kingdom...'")
    print("\n(Continues until <END> token or max length)")


def demo_attention_visualization():
    """Demonstrate attention pattern visualization"""
    section_divider("12. ATTENTION PATTERNS (VISUALIZATION)")
    
    print("Understanding What Attention Learns:\n")
    
    # Simple attention weights for demonstration
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    print(f"Example sentence: {' '.join(sentence)}\n")
    
    # Simulated attention weights when processing "sat"
    print("When processing 'sat', attention weights to:")
    attention_to_sat = [
        ("The", 0.05),
        ("cat", 0.60),
        ("sat", 0.30),
        ("on", 0.03),
        ("the", 0.01),
        ("mat", 0.01),
    ]
    
    print(f"\n{'Token':<10} {'Weight':<10} {'Bar'}")
    print("-" * 40)
    for token, weight in attention_to_sat:
        bar = "█" * int(weight * 50)
        print(f"{token:<10} {weight:<10.2f} {bar}")
    
    print("\nInterpretation:")
    print("  → 'sat' pays most attention to 'cat' (subject-verb relationship)")
    print("  → Also attends to itself")
    print("  → Less attention to other words\n")
    
    print("-" * 70)
    print("\nDifferent heads learn different patterns:")
    print("\nHead 1 (Syntactic): Subject-Verb relationships")
    print("  'cat' → 'sat' (high attention)")
    print("\nHead 2 (Positional): Nearby words")
    print("  Each word → neighbors (high attention)")
    print("\nHead 3 (Semantic): Related meanings")
    print("  'cat' → 'mat' (animals and objects)")
    print("\nHead 4 (Long-range): Beginning and end")
    print("  First word → Last word")
    
    print("\n" + "-" * 70)
    print("This is why multi-head attention is powerful!")
    print("Each head can specialize in different linguistic patterns.")


def demo_training_inference():
    """Demonstrate training vs inference differences"""
    section_divider("13. TRAINING vs INFERENCE")
    
    print("TRAINING (Teacher Forcing):\n")
    
    print("Goal: Translate 'Hello' → 'Bonjour'")
    print("\nAt training time:")
    print("  • We know the correct output")
    print("  • Feed entire target sequence at once")
    print("  • Use masking to prevent seeing future")
    print("  • Compute loss on all positions simultaneously")
    print("  • Much faster (parallel)!\n")
    
    print("Example:")
    print("  Input:  [Hello]")
    print("  Target: [<start> Bonjour <end>]")
    print("  Decoder input: [<start> Bonjour]  (shifted)")
    print("  Decoder target: [Bonjour <end>]   (for loss)")
    print("  ")
    print("  Position 0: Predict 'Bonjour' given '<start>'")
    print("  Position 1: Predict '<end>' given '<start> Bonjour'")
    print("  ")
    print("  ✓ Both predictions done in parallel!")
    
    print("\n" + "=" * 70)
    print("\nINFERENCE (Autoregressive Generation):\n")
    
    print("At inference time:")
    print("  • We don't know the output")
    print("  • Generate one token at a time")
    print("  • Use previous predictions as input")
    print("  • Slower (sequential)\n")
    
    print("Example:")
    print("  Input: [Hello]")
    print("  ")
    print("  Step 1:")
    print("    Decoder input: [<start>]")
    print("    Predict: 'Bonjour'")
    print("  ")
    print("  Step 2:")
    print("    Decoder input: [<start> Bonjour]")
    print("    Predict: '<end>'")
    print("  ")
    print("  Step 3: Stop (saw <end> token)")
    print("  ")
    print("  Final output: Bonjour")
    
    print("\n" + "=" * 70)
    print("\nKey Differences:\n")
    
    print(f"{'Aspect':<25} {'Training':<25} {'Inference'}")
    print("-" * 70)
    print(f"{'Speed':<25} {'Fast (parallel)':<25} {'Slow (sequential)'}")
    print(f"{'Input':<25} {'Full target sequence':<25} {'Generated tokens'}")
    print(f"{'Masking':<25} {'Look-ahead mask':<25} {'Not needed'}")
    print(f"{'Loss':<25} {'All positions':<25} {'N/A'}")


def demo_why_transformers_win():
    """Explain why transformers are so effective"""
    section_divider("14. WHY TRANSFORMERS DOMINATE")
    
    print("Comparison: RNN/LSTM vs Transformer\n")
    
    print(f"{'Feature':<30} {'RNN/LSTM':<25} {'Transformer'}")
    print("=" * 80)
    print(f"{'Processing':<30} {'Sequential':<25} {'Parallel'}")
    print(f"{'Training Speed':<30} {'Slow':<25} {'Fast'}")
    print(f"{'Long-range Dependencies':<30} {'Difficult':<25} {'Easy'}")
    print(f"{'Memory':<30} {'Limited (hidden state)':<25} {'Full sequence'}")
    print(f"{'Parallelization':<30} {'No':<25} {'Yes'}")
    print(f"{'GPU Utilization':<30} {'Poor':<25} {'Excellent'}")
    print(f"{'Vanishing Gradients':<30} {'Yes':<25} {'Minimal'}")
    
    print("\n" + "=" * 80)
    print("\nKey Advantages of Transformers:")
    print("\n1. PARALLELIZATION")
    print("   • Process all tokens simultaneously")
    print("   • Fully utilize modern GPUs")
    print("   • 10-100× faster training than RNNs")
    
    print("\n2. LONG-RANGE DEPENDENCIES")
    print("   • Attention directly connects any two positions")
    print("   • No information bottleneck")
    print("   • Remember information from far away")
    
    print("\n3. FLEXIBILITY")
    print("   • Same architecture for many tasks")
    print("   • Easy to scale (more layers, larger d_model)")
    print("   • Transfer learning works very well")
    
    print("\n4. INTERPRETABILITY")
    print("   • Can visualize attention weights")
    print("   • Understand what model focuses on")
    print("   • Debug and improve more easily")
    
    print("\n5. SCALABILITY")
    print("   • Scales to billions of parameters")
    print("   • Performance improves with size")
    print("   • Compute efficiently on modern hardware")
    
    print("\n" + "=" * 80)
    print("\nResult:")
    print("  Transformers are now the default architecture for:")
    print("    • Natural Language Processing (GPT, BERT, etc.)")
    print("    • Computer Vision (Vision Transformer, CLIP)")
    print("    • Speech Recognition (Whisper)")
    print("    • Protein Folding (AlphaFold)")
    print("    • Code Generation (Codex, Copilot)")
    print("    • Multi-modal AI (GPT-4, Claude)")


def demo_practice_exercises():
    """Provide practice exercises"""
    section_divider("15. PRACTICE EXERCISES")
    
    print("Exercise 1: Attention Computation")
    print("-" * 70)
    print("Given:")
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[2.0, 3.0], [4.0, 5.0]])
    
    print("Q (Query) =")
    print(Q)
    print("\nK (Key) =")
    print(K)
    print("\nV (Value) =")
    print(V)
    
    print("\nTask: Compute attention output")
    print("\nSolution:")
    
    # Compute
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    print(f"\n1. Scores = Q·K^T / √d_k =")
    print(scores)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    print(f"\n2. Attention weights = softmax(scores) =")
    print(weights)
    
    # Output
    output = np.dot(weights, V)
    print(f"\n3. Output = weights·V =")
    print(output)
    
    print("\n" + "=" * 70)
    print("\nExercise 2: Positional Encoding")
    print("-" * 70)
    print("Task: Generate positional encoding for position 0, dimension 4")
    print("\nSolution:")
    
    pos = 0
    d_model = 4
    
    print(f"\nFor position {pos}:")
    for i in range(d_model):
        if i % 2 == 0:
            val = np.sin(pos / (10000 ** (i / d_model)))
            print(f"  PE[{i}] = sin({pos} / 10000^({i}/{d_model})) = {val:.4f}")
        else:
            val = np.cos(pos / (10000 ** ((i-1) / d_model)))
            print(f"  PE[{i}] = cos({pos} / 10000^({i-1}/{d_model})) = {val:.4f}")
    
    print("\n" + "=" * 70)
    print("\nExercise 3: Masking")
    print("-" * 70)
    print("Task: Create look-ahead mask for sequence length 4")
    print("\nSolution:")
    
    seq_len = 4
    mask = np.tril(np.ones((seq_len, seq_len)))
    print("\nLook-ahead mask (1 = allowed, 0 = masked):")
    print(mask.astype(int))
    
    print("\nInterpretation:")
    print("  Row 0: Token 0 can see token 0 only")
    print("  Row 1: Token 1 can see tokens 0, 1")
    print("  Row 2: Token 2 can see tokens 0, 1, 2")
    print("  Row 3: Token 3 can see all tokens")


def main():
    """Main function to run all demonstrations"""
    print("\n" + "★" * 70)
    print("  TRANSFORMERS FOR MACHINE LEARNING")
    print("  From 'Attention Is All You Need' to Modern LLMs")
    print("  Python Implementation with NumPy")
    print("★" * 70)
    
    # Run all demonstrations
    demo_attention_mechanism()
    demo_self_attention()
    demo_multi_head_attention()
    demo_positional_encoding()
    demo_feedforward_network()
    demo_layer_normalization()
    demo_encoder_layer()
    demo_decoder_layer()
    demo_complete_transformer()
    demo_transformer_variants()
    demo_practical_applications()
    demo_attention_visualization()
    demo_training_inference()
    demo_why_transformers_win()
    demo_practice_exercises()
    
    # Final message
    section_divider("COMPLETE!")
    print("You've completed Transformer fundamentals!")
    print("\nKey Takeaways:")
    print("  1. Attention allows focusing on relevant parts of input")
    print("  2. Self-attention lets tokens attend to each other")
    print("  3. Multi-head attention learns different patterns")
    print("  4. Positional encoding provides sequence order")
    print("  5. Residual connections help train deep networks")
    print("  6. Transformers dominate modern NLP and beyond")
    print("\nModern Models Built on Transformers:")
    print("  • GPT-4, Claude (Decoder-only)")
    print("  • BERT, RoBERTa (Encoder-only)")
    print("  • T5, BART (Encoder-Decoder)")
    print("\nNext Steps:")
    print("  1. Implement a mini-transformer from scratch")
    print("  2. Train on a small dataset")
    print("  3. Explore pre-trained models (Hugging Face)")
    print("  4. Fine-tune for your specific task")
    print("  5. Study advanced topics (Flash Attention, efficient transformers)")
    print("\n" + "★" * 70 + "\n")


if __name__ == "__main__":
    main()
