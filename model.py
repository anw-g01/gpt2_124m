import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import inspect
import math
from config import BATCH_SIZE, GRAD_ACCUM_STEPS, ITERATIONS

@dataclass
class GPT2Config:
    block_size: int = 1024     # max sequence (context) length
    vocab_size: int = 50257    # size of token vocabulary --> changed to 50,304 (2^7 * 3 * 131) for efficiency
    n_layer: int = 12          # no. of layers
    n_head: int = 12           # no. of heads
    n_embd: int = 768          # embedding dimensions (64 * 12)

class Attention(nn.Module):
    """
    Combined multi-headed self-attention and causal masking.
    `__call__()` method calculates query, key, values for all heads in batch and transpose
      head forward to be the batch dimension. Where transformer channels, `C` (`n_embd`) = `768` 
    `= n_heads * head_size`, so `head_size = 64` for `GPT-2 (124M)`.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # K, Q, V (batched) projections for all heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # output projection

    def forward(self, x):
        B, T, C = x.shape                                                   # x: [BS, seq_len, n_embd]
        QKV = self.c_attn(x)                                                # --> [BS, seq_len, 3 * n_embd]
        Q, K, V = QKV.split(self.n_embd, dim=2)                             # --> [BS, seq_len, n_embd]
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # --> [BS, n_head, seq_len, head_size]
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # --> ...
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # --> ...
        # ---
        # attn = (Q @ K.transpose(-2, -1)) * (1.0 / np.sqrt(K.shape[-1]))
        # attn = attn.masked_fill(self.bias()[:, :, :T, :T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ V
        # ---
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)         # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)                    # re-assemble all head outputs side by side
        y = self.c_proj(y)                                                  # output projection
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # residual streams
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT2_124M(nn.Module):
    """Architecture of the 124M parameter version of GPT-2."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight sharing scheme
        # for weight initilisation:
        self.res_proj_layers = [block.attn.c_proj for block in self.transformer.h] + \
            [block.mlp.c_proj for block in self.transformer.h]    # residual streams
        self.apply(self._init_weights)   # self.apply() iterates over all sub-modules

    def _init_weights(self, module):
        """
        Initialises the weights for the model layers following GPT-2's initialisation scheme.
        Scales the `c_proj` weights by `1/sqrt(2 * N)`, where `N` is the number of Transformer
        layers. This scaling controls the growth of activations in the residual stream, ensuring
        stable training. The `2 * N` factor arises from each Transformer block contributing two
        sets of weights to the residual stream: one from `attn.c_proj` and one from `mlp.c_proj`.
        """
        std = 0.02      # 0.02 std for GPT-2 (from huggingface)
        if isinstance(module, nn.Linear):
            W, b = module.weight, module.bias
            if module in self.res_proj_layers:
                std *= 1 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(W, mean=0.0, std=std)
            if b is not None:     # if biases are present
                nn.init.zeros_(b)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens: list, targets=None) -> tuple:
        B, T = tokens.shape     # [BS, seq_len]
        assert T <= self.config.block_size, f"Token equence length ({T}) exceeds maximum context length {self.config.block_size}."
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)    # for indexing into Embedding tables
        tok_emb = self.transformer.wte(tokens)      # token embeddings           --> [BS, seq_len, n_embd]
        pos_emb = self.transformer.wpe(pos)         # position embeddings        -->     [seq_len, n_embd]
        x = tok_emb + pos_emb                       # broadcasted addition       --> [BS, seq_len, n_embd]
        for block in self.transformer.h:            # forward all (12) blocks
            x = block(x)
        x = self.transformer.ln_f(x)                # final LayerNorm
        logits = self.lm_head(x)                    # classifier                 --> [BS, seq_len, vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    def configure_optim(self, weight_decay: float, learning_rate: float, device_type: str) -> torch.optim.AdamW:
        """
        Configures and returns an `torch.optim.AdamW()` optimiser with set hyperparameters taken from GPT-3 (125M).
        Implements a parameter grouping strategy to apply `weight_decay` only to `weight` tensors, excluding `bias`
        and `LayerNorm` parameters. The `AdamW()` optimiser toggles on fused kernels when running on `CUDA` devices.
        """
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2 and p.requires_grad]      # weight tensors
        no_decay_params = [p for n, p in self.named_parameters() if p.dim() < 2 and p.requires_grad]    # biases and LayerNorms
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        # -----
        available = "fused" in inspect.signature(torch.optim.AdamW).parameters  # if new "fused" parameter is available
        use_fused = True if device_type == "cuda" else False                    # boolean
        optimiser = torch.optim.AdamW(
            params=optim_groups, lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95), eps=1e-8,    # GPT-3 hyperparameters
            fused=use_fused                 # used fused kernel (accelerate training)
        )
        # --- LOG
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_decay_params = sum(p.numel() for p in decay_params)
        n_no_decay_params = sum(p.numel() for p in no_decay_params)
        pct_decay, pct_no_decay = n_decay_params / n_params * 100, n_no_decay_params / n_params * 100
        print(f"decay params: {n_decay_params:,} ({pct_decay:.1f}%)")
        print(f"no-decay params: {n_no_decay_params:,} ({pct_no_decay:.1f}%)")
        print(f"using fused AdamW: {use_fused} (device={device_type})")
        return optimiser
    
    def sample(self, text: str, n_seqs=5, max_length=30, k=50) -> list:
        """
        Generate new samples of text based on an input.
        -----
        Args:
            - `text` (`str`) - input string of text to feed into the model.
        
        Keyword args:
            - `n_seqs` (`int`) - number of return sequences to output.
            - `max_length` (`int`) - token length of each return sequence.
            - `k` (`int`) - limits sampling to the top `k` most probable tokens at each step.
       
       Returns:
            - `dict` - dictionary of `n_seqs` generated text samples.
        """
        enc = tiktoken.get_encoding("gpt2")    # use the GPT-2 tokenizer
        tokens = torch.tensor(
            data=enc.encode(text),
            dtype=torch.long,
            device=torch.device("cuda")
        ).unsqueeze(0).repeat(n_seqs, 1)
        i, n = 1, max_length - x.shape[1]    # starting token idx, tokens left to generate
        while x.shape[1] < max_length:
            pct = (i / n) * 100
            print(f"\rsampling {i}/{n} ({pct:.0f}%)", end="")
            with torch.no_grad():
                logits, _ = self(x)         # model forward pass --> [n_seqs, curr_seq_length, vocab_size]
                logits = logits[:, -1, :]   # take logits of last token --> [n_seqs, vocab_size]
                probs = F.softmax(logits, dim=-1)
                # filter to sample from the top 'k' most probable tokens:
                topk_probs, topk_ids = torch.topk(probs, k, dim=-1)     # --> [n_seqs, k]
                idx = torch.multinomial(topk_probs, num_samples=1)      # --> [n_seqs, 1]
                # map sampled indices back to original vocab indicies:
                x_col = torch.gather(topk_ids, dim=-1, index=idx)       # --> [n_seqs, 1]
                x = torch.cat((x, x_col), dim=1)    # concatenate newly sampled tokens --> [n_seqs, curr_seq_length + 1]
            i += 1
        # final x.shape --> [n_seqs, max_length]
        out = {}
        for i in range(x):
            out[i] = x[i:max_length].tolist()
        return out
    
    def save(self, file_name=None):
        """
        Save and download model's weights in a PyTorch `.pth` file.
        """
        if file_name is None:
            batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS
            file_name = f"GPT2_124M_BS({batch_size})_ITER({ITERATIONS})"
        torch.save(self.state_dict(), f"{file_name}.pth")
        print(f"model weights saved as: {file_name}")
