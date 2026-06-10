"""
Razonamiento Latente: Abstract-CoT y VSA

Implementa "pensar sin palabras" mediante:
- Tokens abstractos como bloc de notas latente
- Álgebra hiperdimensional con VSA
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
    """Configuración para Abstract Chain-of-Thought"""
    num_abstract_tokens: int = 256  # Tamaño del vocabulario reservado
    d_model: int = 768
    max_reasoning_steps: int = 32
    binding_dim: int = 10000  # Dimensión hiperdimensional VSA
    use_vsa: bool = True
    temperature: float = 0.8
    top_k: int = 10


class CircularConvolution(nn.Module):
    """
    Operación de Binding (⊛) mediante convolución circular
    Vincula roles y entidades: "Agente" ⊛ "Humano"
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Convolución circular entre dos vectores hiperdimensionales
        
        Args:
            a, b: (..., dim)
        Returns:
            bound: (..., dim) a ⊛ b
        """
        # FFT-based circular convolution
        a_fft = torch.fft.fft(a.float(), dim=-1)
        b_fft = torch.fft.fft(b.float(), dim=-1)
        
        # Multiplicación en frecuencia
        bound_fft = a_fft * b_fft
        
        # IFFT para obtener resultado
        bound = torch.fft.ifft(bound_fft, dim=-1).real
        
        return bound
    
    def inverse(self, bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Desvinculación: dado a ⊛ b y a, recuperar b
        """
        a_fft = torch.fft.fft(a.float(), dim=-1)
        bound_fft = torch.fft.fft(bound.float(), dim=-1)
        
        # División en frecuencia
        b_fft = bound_fft / (a_fft + 1e-10)
        
        b = torch.fft.ifft(b_fft, dim=-1).real
        return b


class HyperdimensionalEncoder(nn.Module):
    """
    Codificador de símbolos a espacio hiperdimensional
    """
    
    def __init__(self, vocab_size: int, dim: int, num_positions: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_positions = num_positions
        
        # Vectores base aleatorios para cada símbolo
        self.symbol_vectors = nn.Parameter(
            torch.randn(vocab_size, dim) / math.sqrt(dim)
        )
        
        # Vectores de posición
        self.position_vectors = nn.Parameter(
            self._generate_position_vectors(num_positions, dim)
        )
    
    def _generate_position_vectors(self, n: int, d: int) -> torch.Tensor:
        """Generar vectores de posición usando patrones de fase"""
        positions = torch.arange(n).unsqueeze(1).float()
        dims = torch.arange(d).unsqueeze(0).float()
        
        # Patrones de fase
        phase = positions * (2 ** (-dims / d))
        
        # Codificación
        vectors = torch.where(
            dims % 2 == 0,
            torch.sin(phase),
            torch.cos(phase)
        )
        
        return vectors / math.sqrt(d)
    
    def encode(self, symbols: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Codificar símbolos a espacio hiperdimensional
        
        Args:
            symbols: (B, L) índices de símbolos
            positions: (B, L) posiciones (opcional)
        Returns:
            encoded: (B, L, dim) vectores hiperdimensionales
        """
        batch_size, seq_len = symbols.shape
        
        # Lookup de vectores de símbolo
        symbol_vecs = F.embedding(symbols, self.symbol_vectors)  # (B, L, dim)
        
        # Añadir posición si se proporciona
        if positions is not None:
            pos_vecs = F.embedding(positions, self.position_vectors)
            symbol_vecs = symbol_vecs + pos_vecs
        
        # Binarizar (opcional para VSA clásica)
        # symbol_vecs = torch.where(symbol_vecs > 0, 1.0, -1.0)
        
        return symbol_vecs


class VSAModule(nn.Module):
    """
    Módulo de Vector Symbolic Architecture
    
    Operaciones:
    - Binding (⊛): Vincula roles y entidades
    - Bundling (+): Superposición de hechos
    """
    
    def __init__(self, config: AbstractCoTConfig):
        super().__init__()
        self.config = config
        self.dim = config.binding_dim
        
        # Codificadores
        self.entity_encoder = HyperdimensionalEncoder(
            vocab_size=10000,  # Tamaño de vocabulario de entidades
            dim=self.dim,
            num_positions=128
        )
        
        self.role_encoder = HyperdimensionalEncoder(
            vocab_size=100,  # Roles: agente, paciente, acción, etc.
            dim=self.dim,
            num_positions=10
        )
        
        # Operación de binding
        self.binding = CircularConvolution(self.dim)
        
        # Red de atención sobre superposiciones
        self.attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=8,
            batch_first=True
        )
        
        # Decodificador
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.GELU(),
            nn.Linear(self.dim // 2, config.d_model)
        )
    
    def bind(self, role: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
        """
        Binding: role ⊛ entity
        
        Args:
            role: (B, dim) vector de rol
            entity: (B, dim) vector de entidad
        Returns:
            bound: (B, dim) vector vinculado
        """
        return self.binding(role, entity)
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Bundling: Superposición de múltiples vectores
        v_sum = v1 + v2 + ... + vn
        """
        if not vectors:
            return torch.zeros(1, self.dim)
        
        stacked = torch.stack(vectors, dim=1)  # (B, N, dim)
        
        # Sumar con normalización
        bundled = stacked.sum(dim=1)  # (B, dim)
        bundled = F.normalize(bundled, p=2, dim=-1) * math.sqrt(self.dim)
        
        return bundled
    
    def unbind_and_query(self, 
                         memory: torch.Tensor, 
                         query_role: torch.Tensor) -> torch.Tensor:
        """
        Desvincular y consultar en memoria
        
        Args:
            memory: (B, N, dim) memoria superpuesta
            query_role: (B, dim) rol a consultar
        Returns:
            result: (B, dim) entidad recuperada
        """
        batch_size = memory.size(0)
        num_facts = memory.size(1)
        
        # Expandir query para todos los hechos
        query_expanded = query_role.unsqueeze(1).expand(-1, num_facts, -1)
        
        # Desvincular cada hecho
        results = []
        for i in range(num_facts):
            unbound = self.binding.inverse(memory[:, i], query_expanded[:, i])
            results.append(unbound)
        
        # Combinar resultados
        result = torch.stack(results, dim=1).mean(dim=1)
        
        # Atención sobre el resultado
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
        Codificar una estructura compleja (ej: "Juan come manzana")
        """
        # Convertir a índices
        role_indices = torch.tensor(
            [hash(r) % 100 for r in roles],
            device=device
        ).unsqueeze(0)
        entity_indices = torch.tensor(
            [hash(e) % 10000 for e in entities],
            device=device
        ).unsqueeze(0)
        
        # Codificar
        role_vecs = self.role_encoder.encode(role_indices)
        entity_vecs = self.entity_encoder.encode(entity_indices)
        
        # Vincular cada par
        bound_vecs = []
        for r, e in zip(role_vecs[0], entity_vecs[0]):
            bound = self.bind(r.unsqueeze(0), e.unsqueeze(0))
            bound_vecs.append(bound)
        
        # Bundling final
        memory = self.bundle(bound_vecs)
        
        return memory
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesar entrada con operaciones VSA
        
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Proyectar a espacio hiperdimensional
        # Simplificación: usar proyección lineal
        proj = nn.Linear(x.size(-1), self.dim, device=x.device)
        x_hd = proj(x)  # (B, L, dim)
        
        # Aplicar binding con roles posicionales
        role_indices = torch.arange(seq_len, device=x.device) % 100
        role_vecs = self.role_encoder.encode(
            role_indices.unsqueeze(0).expand(batch_size, -1)
        )
        
        # Vincular cada posición con su rol
        bound = []
        for i in range(seq_len):
            b = self.bind(x_hd[:, i], role_vecs[:, i])
            bound.append(b)
        
        # Bundling de toda la secuencia
        memory = self.bundle(bound)
        
        # Expandir de vuelta a longitud de secuencia
        memory_expanded = memory.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Decodificar
        output = self.decoder(memory_expanded)
        
        return output


class AbstractTokenizer:
    """
    Tokenizador para vocabulario de tokens abstractos
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
        """Obtener token para tipo de pensamiento"""
        return self.special_tokens.get(f'<{thought_type}>', 0)
    
    def get_abstract_token(self, index: int) -> int:
        """Obtener token abstracto por índice"""
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
    
    "Pensar sin palabras" usando tokens abstractos como bloc de notas latente.
    Eficiencia: medida dinámicamente vía get_efficiency_ratio() — no hardcodeada.
    """
    
    def __init__(self, config: AbstractCoTConfig):
        super().__init__()
        self.config = config
        
        # Tokenizador abstracto
        self.tokenizer = AbstractTokenizer(config.num_abstract_tokens)
        
        # Embedding de tokens abstractos
        self.abstract_embed = nn.Embedding(
            self.tokenizer.vocab_size,
            config.d_model
        )
        
        # Controlador de razonamiento
        self.reasoning_controller = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Generador de tokens abstractos
        self.token_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, self.tokenizer.vocab_size)
        )
        
        # Módulo VSA
        if config.use_vsa:
            self.vsa = VSAModule(config)
        else:
            self.vsa = None
        
        # Atención cruzada: razonamiento -> output
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Capa de salida
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        
        # Gate para mezclar información
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )
    
    def generate_abstract_sequence(self, 
                                   context: torch.Tensor,
                                   num_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generar secuencia de razonamiento abstracto
        
        Args:
            context: (B, L, d_model) contexto de entrada
            num_steps: Número de pasos de razonamiento (default: max_reasoning_steps)
        Returns:
            abstract_tokens: (B, num_steps) índices de tokens
            reasoning_states: (B, num_steps, d_model) estados de razonamiento
        """
        if num_steps is None:
            num_steps = self.config.max_reasoning_steps
        
        batch_size = context.size(0)
        device = context.device
        
        # Estado inicial: promedio del contexto
        h0 = context.mean(dim=1, keepdim=True).transpose(0, 1)  # (1, B, d)
        h0 = h0.expand(2, -1, -1).contiguous()  # (num_layers, B, d)
        c0 = torch.zeros_like(h0)
        
        hidden = (h0, c0)
        
        # Token inicial
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
            
            # Paso LSTM
            lstm_out, hidden = self.reasoning_controller(token_emb, hidden)
            state = lstm_out.squeeze(1)  # (B, d)
            states.append(state)
            
            # Generar siguiente token
            logits = self.token_generator(state)  # (B, vocab_size)
            
            # Sample con temperatura
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, self.config.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Sample
            sampled_idx = torch.multinomial(top_k_probs, 1)
            current_token = top_k_indices.gather(1, sampled_idx)  # (B, 1)
            
            tokens.append(current_token.squeeze(1))
            
            # Detener si se genera token de fin
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
        Forward pass con Abstract-CoT
        
        Args:
            input_ids: (B, L)
            context: (B, L, d_model) representaciones del encoder
            use_reasoning: Si usar razonamiento abstracto
        Returns:
            dict con output, tokens abstractos, estados de razonamiento
        """
        if not use_reasoning or not self.training:
            # Modo inference sin razonamiento explícito
            return {
                'output': context,
                'abstract_tokens': None,
                'reasoning_states': None
            }
        
        # Generar secuencia de razonamiento abstracto
        abstract_tokens, reasoning_states = self.generate_abstract_sequence(context)
        
        # Procesar con VSA si está habilitado
        if self.vsa is not None:
            reasoning_states = self.vsa(reasoning_states)
        
        # Atención cruzada: razonamiento enriquece contexto
        context_enhanced, _ = self.cross_attention(
            query=context,
            key=reasoning_states,
            value=reasoning_states
        )
        
        # Gate para mezclar
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
        Calcular ratio de eficiencia: tokens verbales / tokens abstractos.
        Usa el measured ratio de get_efficiency_ratio(), no un valor hardcodeado.
        """
        verbal_tokens = input_ids.size(1)
        abstract_count = abstract_tokens.size(1) if abstract_tokens is not None else 1
        
        return get_efficiency_ratio(verbal_tokens, abstract_count)
    
    def decode_abstract(self, tokens: torch.Tensor) -> List[str]:
        """
        Decodificar tokens abstractos a representación legible (para debugging)
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
