"""
Latent Reasoning: Abstract-CoT and VSA

Implements "think without words" via:
- Abstract tokens as latent scratchpad
- Hyperdimensional algebra with VSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import math
import random


@dataclass
class AbstractCoTConfig:
    """Configuration for Abstract Chain-of-Thought"""
    num_abstract_tokens: int = 256  # Reserved vocabulary size
    d_model: int = 768
    max_reasoning_steps: int = 32
    binding_dim: int = 10000  # Hyperdimensional VSA dimension
    use_vsa: bool = True
    temperature: float = 0.8
    top_k: int = 10


class CircularConvolution(nn.Module):
    """
    Binding operation (⊛) via circular convolution
    Binds roles and entities: "Agent" ⊛ "Human"
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution between two hyperdimensional vectors
        
        Args:
            a, b: (..., dim)
        Returns:
            bound: (..., dim) a ⊛ b
        """
        # FFT-based circular convolution
        a_fft = torch.fft.fft(a.float(), dim=-1)
        b_fft = torch.fft.fft(b.float(), dim=-1)
        
        # Frequency-domain multiplication
        bound_fft = a_fft * b_fft
        
        # IFFT to obtain result
        bound = torch.fft.ifft(bound_fft, dim=-1).real
        
        return bound
    
    def inverse(self, bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Unbinding: given a ⊛ b and a, recover b
        """
        a_fft = torch.fft.fft(a.float(), dim=-1)
        bound_fft = torch.fft.fft(bound.float(), dim=-1)
        
        # Frequency-domain division
        b_fft = bound_fft / (a_fft + 1e-10)
        
        b = torch.fft.ifft(b_fft, dim=-1).real
        return b


class HyperdimensionalEncoder(nn.Module):
    """
    Encoder of symbols to hyperdimensional space
    """
    
    def __init__(self, vocab_size: int, dim: int, num_positions: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_positions = num_positions
        
        # Random base vectors for each symbol
        self.symbol_vectors = nn.Parameter(
            torch.randn(vocab_size, dim) / math.sqrt(dim)
        )
        
        # Position vectors
        self.position_vectors = nn.Parameter(
            self._generate_position_vectors(num_positions, dim)
        )
    
    def _generate_position_vectors(self, n: int, d: int) -> torch.Tensor:
        """Generate position vectors using phase patterns"""
        positions = torch.arange(n).unsqueeze(1).float()
        dims = torch.arange(d).unsqueeze(0).float()
        
        # Phase patterns
        phase = positions * (2 ** (-dims / d))
        
        # Encoding
        vectors = torch.where(
            dims % 2 == 0,
            torch.sin(phase),
            torch.cos(phase)
        )
        
        return vectors / math.sqrt(d)
    
    def encode(self, symbols: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode symbols to hyperdimensional space
        
        Args:
            symbols: (B, L) symbol indices
            positions: (B, L) positions (optional)
        Returns:
            encoded: (B, L, dim) hyperdimensional vectors
        """
        batch_size, seq_len = symbols.shape
        
        # Symbol vector lookup
        symbol_vecs = F.embedding(symbols, self.symbol_vectors)  # (B, L, dim)
        
        # Add position if provided
        if positions is not None:
            pos_vecs = F.embedding(positions, self.position_vectors)
            symbol_vecs = symbol_vecs + pos_vecs
        
        # Binarize (optional for classical VSA)
        # symbol_vecs = torch.where(symbol_vecs > 0, 1.0, -1.0)
        
        return symbol_vecs


class VSAModule(nn.Module):
    """
    Vector Symbolic Architecture Module
    
    Operations:
    - Binding (⊛): Binds roles and entities
    - Bundling (+): Fact superposition
    """
    
    def __init__(self, config: AbstractCoTConfig):
        super().__init__()
        self.config = config
        self.dim = config.binding_dim
        
        # Codificadores
        self.entity_encoder = HyperdimensionalEncoder(
            vocab_size=10000,  # Entity vocabulary size
            dim=self.dim,
            num_positions=128
        )
        
        self.role_encoder = HyperdimensionalEncoder(
            vocab_size=100,  # Roles: agent, patient, action, etc.
            dim=self.dim,
            num_positions=10
        )
        
        # Binding operation
        self.binding = CircularConvolution(self.dim)
        
        # Attention network over superpositions
        self.attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=8,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.GELU(),
            nn.Linear(self.dim // 2, config.d_model)
        )
        
        # Input projection (registered submodule for gradient flow)
        self.input_proj = nn.Linear(config.d_model, self.dim) if config.d_model != self.dim else nn.Identity()
    
    def bind(self, role: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
        """
        Binding: role ⊛ entity
        
        Args:
            role: (B, dim) role vector
            entity: (B, dim) entity vector
        Returns:
            bound: (B, dim) bound vector
        """
        return self.binding(role, entity)
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Bundling: Superposition of multiple vectors
        v_sum = v1 + v2 + ... + vn
        """
        if not vectors:
            return torch.zeros(1, self.dim)
        
        stacked = torch.stack(vectors, dim=1)  # (B, N, dim)
        
        # Sum with normalization
        bundled = stacked.sum(dim=1)  # (B, dim)
        bundled = F.normalize(bundled, p=2, dim=-1) * math.sqrt(self.dim)
        
        return bundled
    
    def unbind_and_query(self, 
                         memory: torch.Tensor, 
                         query_role: torch.Tensor) -> torch.Tensor:
        """
        Unbind and query memory
        
        Args:
            memory: (B, N, dim) superimposed memory
            query_role: (B, dim) role to query
        Returns:
            result: (B, dim) recovered entity
        """
        batch_size = memory.size(0)
        num_facts = memory.size(1)
        
        # Expand query for all facts
        query_expanded = query_role.unsqueeze(1).expand(-1, num_facts, -1)
        
        # Unbind each fact
        results = []
        for i in range(num_facts):
            unbound = self.binding.inverse(memory[:, i], query_expanded[:, i])
            results.append(unbound)
        
        # Combine results
        result = torch.stack(results, dim=1).mean(dim=1)
        
        # Attention on result
        result, _ = self.attention(
            result.unsqueeze(1),
            memory,
            memory
        )
        
        return result.squeeze(1)
    
    def encode_structure(self, 
                         roles: List[str], 
                         entities: List[str],
                         device: str = "cuda") -> torch.Tensor:
        """
        Encode a complex structure (e.g., "John eats apple")
        """
        # Convert to indices (deterministic hash via hashlib, not Python's salted hash)
        import hashlib
        def deterministic_hash(s: str, mod: int) -> int:
            return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod
        
        role_indices = torch.tensor(
            [deterministic_hash(r, 100) for r in roles],
            device=device
        ).unsqueeze(0)
        entity_indices = torch.tensor(
            [deterministic_hash(e, 10000) for e in entities],
            device=device
        ).unsqueeze(0)
        
        # Encode
        role_vecs = self.role_encoder.encode(role_indices)
        entity_vecs = self.entity_encoder.encode(entity_indices)
        
        # Bind each pair
        bound_vecs = []
        for r, e in zip(role_vecs[0], entity_vecs[0]):
            bound = self.bind(r.unsqueeze(0), e.unsqueeze(0))
            bound_vecs.append(bound)
        
        # Final bundling
        memory = self.bundle(bound_vecs)
        
        return memory
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input with VSA operations
        
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to HD space using registered projection
        x_hd = self.input_proj(x)  # (B, L, dim)
        
        # Apply binding with positional roles
        role_indices = torch.arange(seq_len, device=x.device) % 100
        role_vecs = self.role_encoder.encode(
            role_indices.unsqueeze(0).expand(batch_size, -1)
        )
        
        # Bind each position to its role
        bound = []
        for i in range(seq_len):
            b = self.bind(x_hd[:, i], role_vecs[:, i])
            bound.append(b)
        
        # Bundling of the entire sequence
        memory = self.bundle(bound)
        
        # Expand back to sequence length
        memory_expanded = memory.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Decode
        output = self.decoder(memory_expanded)
        
        return output


class AbstractTokenizer:
    """
    Tokenizer for abstract token vocabulary
    """
    
    def __init__(self, num_tokens: int = 256):
        self.num_tokens = num_tokens
        self.special_tokens = {
            '<start>': num_tokens,
            '<end>': num_tokens + 1,
            '<think>': num_tokens + 2,
            '<reason>': num_tokens + 3,
            '<infer>': num_tokens + 4
        }
        self.vocab_size = num_tokens + len(self.special_tokens)
    
    def encode_thought(self, thought_type: str) -> int:
        """Get token for thought type"""
        return self.special_tokens.get(f'<{thought_type}>', 0)
    
    def get_abstract_token(self, index: int) -> int:
        """Get abstract token by index"""
        return index % self.num_tokens


def get_efficiency_ratio(num_verbal_tokens: int, num_abstract_tokens: int) -> float:
    """
    Calculate the TRUE measured efficiency ratio of Abstract-CoT vs verbal CoT.
    
    This function returns a dynamic, measured ratio — NOT a hardcoded constant.
    The previous "11.6x" claim was a hardcoded value with no empirical basis.
    
    Args:
        num_verbal_tokens: Number of tokens a verbal CoT would produce.
        num_abstract_tokens: Number of abstract tokens actually used.
    
    Returns:
        ratio: num_verbal_tokens / num_abstract_tokens (measured at runtime).
            Returns 0.0 if either input is <= 0.
    """
    if num_verbal_tokens <= 0 or num_abstract_tokens <= 0:
        return 0.0
    return num_verbal_tokens / num_abstract_tokens


class AbstractCoT(nn.Module):
    """
    Abstract Chain-of-Thought
    
    "Think without words" using abstract tokens as a latent scratchpad.
    Efficiency: measured dynamically via get_efficiency_ratio() — not hardcoded.
    """
    
    def __init__(self, config: AbstractCoTConfig):
        super().__init__()
        self.config = config
        
        # Abstract tokenizer
        self.tokenizer = AbstractTokenizer(config.num_abstract_tokens)
        
        # Abstract token embedding
        self.abstract_embed = nn.Embedding(
            self.tokenizer.vocab_size,
            config.d_model
        )
        
        # Reasoning controller
        self.reasoning_controller = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Abstract token generator
        self.token_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, self.tokenizer.vocab_size)
        )
        
        # VSA module
        if config.use_vsa:
            self.vsa = VSAModule(config)
        else:
            self.vsa = None
        
        # Cross attention: reasoning -> output
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Output layer
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        
        # Gate to mix information
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )
    
    def generate_abstract_sequence(self, 
                                   context: torch.Tensor,
                                   num_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate abstract reasoning sequence
        
        Args:
            context: (B, L, d_model) input context
            num_steps: Number of reasoning steps (default: max_reasoning_steps)
        Returns:
            abstract_tokens: (B, num_steps) token indices
            reasoning_states: (B, num_steps, d_model) reasoning states
        """
        if num_steps is None:
            num_steps = self.config.max_reasoning_steps
        
        batch_size = context.size(0)
        device = context.device
        
        # Initial state: context average
        h0 = context.mean(dim=1, keepdim=True).transpose(0, 1)  # (1, B, d)
        h0 = h0.expand(2, -1, -1).contiguous()  # (num_layers, B, d)
        c0 = torch.zeros_like(h0)
        
        hidden = (h0, c0)
        
        # Initial token
        current_token = torch.full(
            (batch_size, 1),
            self.tokenizer.encode_thought('think'),
            dtype=torch.long,
            device=device
        )
        
        tokens = [current_token.squeeze(1)]
        states = []
        
        for step in range(num_steps):
            # Embed token actual
            token_emb = self.abstract_embed(current_token.squeeze(1))  # (B, d)
            token_emb = token_emb.unsqueeze(1)  # (B, 1, d)
            
            # Step LSTM
            lstm_out, hidden = self.reasoning_controller(token_emb, hidden)
            state = lstm_out.squeeze(1)  # (B, d)
            states.append(state)
            
            # Generar siguiente token
            logits = self.token_generator(state)  # (B, vocab_size)
            
            # Sample with temperature
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, self.config.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Sample
            sampled_idx = torch.multinomial(top_k_probs, 1)
            current_token = top_k_indices.gather(1, sampled_idx)  # (B, 1)
            
            tokens.append(current_token.squeeze(1))
            
            # Stop if end token generated
            if (current_token == self.tokenizer.encode_thought('end')).all():
                break
        
        abstract_tokens = torch.stack(tokens[:-1], dim=1)  # (B, num_steps)
        reasoning_states = torch.stack(states, dim=1)  # (B, num_steps, d)
        
        return abstract_tokens, reasoning_states
    
    def forward(self, 
                input_ids: torch.Tensor,
                context: torch.Tensor,
                use_reasoning: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Abstract-CoT
        
        Args:
            input_ids: (B, L)
            context: (B, L, d_model) encoder representations
            use_reasoning: Whether to use abstract reasoning
        Returns:
            dict with output, abstract tokens, reasoning states
        """
        if not use_reasoning or not self.training:
            # Inference mode without explicit reasoning
            return {
                'output': context,
                'abstract_tokens': None,
                'reasoning_states': None
            }
        
        # Generar secuencia de razonamiento abstracto
        abstract_tokens, reasoning_states = self.generate_abstract_sequence(context)
        
        # Process with VSA if enabled
        if self.vsa is not None:
            reasoning_states = self.vsa(reasoning_states)
        
        # Cross attention: reasoning enriches context
        context_enhanced, _ = self.cross_attention(
            query=context,
            key=reasoning_states,
            value=reasoning_states
        )
        
        # Gate to mix
        gate_input = torch.cat([context, context_enhanced], dim=-1)
        g = self.gate(gate_input)
        
        output = g * context_enhanced + (1 - g) * context
        output = self.output_proj(output)
        
        return {
            'output': output,
            'abstract_tokens': abstract_tokens,
            'reasoning_states': reasoning_states,
            'efficiency_ratio': self._compute_efficiency_ratio(input_ids, abstract_tokens)
        }
    
    def _compute_efficiency_ratio(self, 
                                  input_ids: torch.Tensor, 
                                  abstract_tokens: torch.Tensor) -> float:
        """
        Calculate efficiency ratio: verbal tokens / abstract tokens.
        Uses the measured ratio from get_efficiency_ratio(), not a hardcoded value.
        """
        verbal_tokens = input_ids.size(1)
        abstract_count = abstract_tokens.size(1) if abstract_tokens is not None else 1
        
        return get_efficiency_ratio(verbal_tokens, abstract_count)
    
    def decode_abstract(self, tokens: torch.Tensor) -> List[str]:
        """
        Decode abstract tokens to readable representation (for debugging)
        """
        decoded = []
        for seq in tokens:
            seq_decoded = []
            for t in seq.tolist():
                if t < self.config.num_abstract_tokens:
                    seq_decoded.append(f"[A{t}]")
                else:
                    for name, idx in self.tokenizer.special_tokens.items():
                        if t == idx:
                            seq_decoded.append(name)
                            break
            decoded.append(" ".join(seq_decoded))
        return decoded
