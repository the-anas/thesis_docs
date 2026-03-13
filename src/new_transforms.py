import torch
import torch.nn as nn
from compressai.models.utils import conv, deconv
from compressai.layers import GDN
import torch.nn.functional as F

# from new_utils import patchify, embed_image, unpatchify


# [X] write encoder

class Encoder_CrossAttention(nn.Module):
    def __init__(self, N, M, K,  num_heads: int = 8, attn_dim: int | None = None,  **kwargs):
        super().__init__(**kwargs)


        # 1) Local conv stack (same spirit as your original g_a)
        self.local = nn.Sequential(
            conv(3, N),#8, 16
            GDN(N),
            conv(N, N), #4, 8
            GDN(N),
            # conv(N, N), #2, 4
            # GDN(N),
            conv(N, M, stride=1, kernel_size=3),  # -> (B*P, M, h', w')
        )
        # [] make convolution a bit less deep



        d = attn_dim if attn_dim is not None else M
        assert d % num_heads == 0, "attn_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(M, d)
        self.k_proj = nn.Linear(K, d) # attempted fix for avg_pool K*16*16
        self.v_proj = nn.Linear(K, d) # attempted fix for avg_pool K*16*16
        self.out_proj = nn.Linear(d, M)

        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)


    def forward(self, x_p, y_g):

        # [] YOU NEED BETTER SHAPING APPROACHES AND TO VERIFY THIS
        
        B, P, C, Hp, Wp = x_p.shape


        x_flat = x_p.reshape(B * P, C, Hp, Wp)
        # print("x_flat shape", x_flat.shape)

        # ---- Local features (B*P, M, h', w') ----
        y_local = self.local(x_flat)
        BP, M, h, w = y_local.shape  # BP == B*P
        # print("shape of y_locoal", y_local.shape)    

        # print("x_g shape", x_p.shape)
        _,K,h_,w_ = y_g.shape
        # print("y_g shape", y_g.shape)

        # ---- Turn feature map into query tokens: (B*P, L, M) where L=h*w ----
        # tokens correspond to spatial locations inside the patch latent
        q = y_local.flatten(2).transpose(1, 2)  # (BP, L, M)

        # print("shape of q", q.shape)
        # ---- Global context as 1 token per patch: (B*P, 1, K) ----
        # kv = y_g.reshape(B * P, h_*w_, K) # attempted fix for avg_pool K*16*16
        kv = y_g.flatten(2).transpose(1, 2).contiguous()  # (BP, h_*w_, K)
        # print("KV PASSED")
        # ---- Project + layernorm ----
        q = self.ln_q(self.q_proj(q))         # (BP, L, d)
        k = self.ln_kv(self.k_proj(kv))       # (BP, 1, d)
        v = self.v_proj(kv)                   # (BP, 1, d)

        # ---- Cross-attention: patch tokens attend to global token(s) ----
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False)  # (BP, L, d)

        # ---- Project back to M and add residual to local tokens ----
        attn_out = self.out_proj(attn_out)    # (BP, L, M)
        
        # q_fused = q.new_zeros(BP, h * w, M)    # just to be explicit about dtype/device
        # skip connection for cross attention, standard practice to have
    
        fused = attn_out + (y_local.flatten(2).transpose(1, 2))  # residual in M-space

        # [] shoud you add ffn layer after attention
        #    fused = fused + self.ffn(fused)                # (BP, L, M)

        # print("shape fused", fused.shape)

        # ---- Back to feature map: (BP, M, h, w) ----
        y = fused.transpose(1, 2).reshape(BP, M, h, w)
        
        return y
    



class Decoder_CrossAttention(nn.Module):
    """
    g_s(y_hat, y_g): cross-attention in latent space first, then deconv stack.
    """
    def __init__(self, N: int, M: int,  K: int, num_heads: int = 8, attn_dim: int | None = None):
        super().__init__()
        self.M = int(M)
        self.N = int(N)
        self.K = int(K)

        d = attn_dim if attn_dim is not None else M
        assert d % num_heads == 0, "attn_dim must be divisible by num_heads"

        # Cross-attn in latent space (tokens of size M)
        self.q_proj = nn.Linear(M, d)
        self.k_proj = nn.Linear(K, d)
        self.v_proj = nn.Linear(K, d)
        self.out_proj = nn.Linear(d, M)

        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True)

        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.LayerNorm(M),
            nn.Linear(M, 4 * M),
            nn.GELU(),
            nn.Linear(4 * M, M),
        )

        # Then the usual synthesis transform (deconv stack)
        self.decode = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            # GDN(N, inverse=True),
            # deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3, stride=1, kernel_size=3),
        )

    def forward(self, y_hat: torch.Tensor, y_g: torch.Tensor) -> torch.Tensor:
        
        """
        y_hat: (BP, M, h, w)
        y_g:   (B, K) or (B, P, K)  -> broadcast to (BP, K)

        returns:
          x_hat_patches: (BP, 3, Hp, Wp)
        """

        BP, M, h, w = y_hat.shape
        # print(f"\n\nCurrently in decoder\n{y_g.shape}")

        # kv = y_g.reshape(BP, 1, self.K)  # (BP, 1, K)
        kv = y_g.flatten(2).transpose(1, 2).contiguous()  # (BP, h_*w_, K)

        # Latent tokens as queries: (BP, L, M) where L=h*w
        q_tokens = y_hat.flatten(2).transpose(1, 2)  # (BP, L, M)

        # Projections + norms
        q = self.ln_q(self.q_proj(q_tokens))   # (BP, L, d)
        k = self.ln_kv(self.k_proj(kv))        # (BP, 1, d)
        v = self.v_proj(kv)                    # (BP, 1, d)

        # Cross-attention (latent attends to global)
        attn_out, _ = self.attn(q, k, v, need_weights=False)  # (BP, L, d)
        attn_out = self.out_proj(attn_out)                    # (BP, L, M)

        # Residual + FFN (Transformer block style)
        fused = q_tokens + attn_out
        
        # [] to enbale if planning on using mlp later on
        # fused = fused + self.ffn(fused)

        # Back to feature map (BP, M, h, w)
        y_fused = fused.transpose(1, 2).reshape(BP, M, h, w)

        # Decode to pixel patches
        x_hat = self.decode(y_fused)  # (BP, 3, Hp, Wp)

        return x_hat



# new_transforms_v2.py




class BahdanauCrossAttention(nn.Module):
    """
    Additive (Bahdanau-style) cross-attention.

    Each query token (from the patch latent) attends over all key-value
    tokens (from the patch embedding) via a learned MLP alignment score:

        e_ij = v^T * tanh(W_q * q_i  +  W_k * k_j)   [scalar per (i,j) pair]
        alpha_ij = softmax_j(e_ij)
        out_i = sum_j alpha_ij * v_j

    Args:
        q_dim:   dimensionality of query vectors  (M, the latent channel dim)
        kv_dim:  dimensionality of key/value vectors  (K, the embedding dim)
        attn_dim: hidden size of the alignment MLP (defaults to q_dim)
    """
    def __init__(self, q_dim: int, kv_dim: int, attn_dim: int | None = None):
        super().__init__()
        d = attn_dim if attn_dim is not None else q_dim

        self.W_q = nn.Linear(q_dim,  d, bias=False)
        self.W_k = nn.Linear(kv_dim, d, bias=False)
        self.v   = nn.Linear(d, 1,    bias=False) # energy

        # value projection: keys and values can have different dims
        self.W_v = nn.Linear(kv_dim, q_dim, bias=False)

        # [] why is out proj here and is it being used in gpt??
        self.out_proj = nn.Linear(q_dim, q_dim)

    # [] takes in args completely different to gpt??
    def forward(
        self,
        q: torch.Tensor,   # (B, L_q, q_dim)
        kv: torch.Tensor,  # (B, L_k, kv_dim)
    ): # -> torch.Tensor:     # (B, L_q, q_dim)

        # --- alignment scores ---
        # W_q(q):  (B, L_q, d)   W_k(kv): (B, L_k, d)
        # unsqueeze to broadcast over the opposing sequence dim
        # [] why the unsqueeze exactly??
        q_proj  = self.W_q(q).unsqueeze(2)    # (B, L_q, 1,   d)
        k_proj  = self.W_k(kv).unsqueeze(1)   # (B, 1,   L_k, d)

        energy  = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # (B, L_q, L_k)
        alpha   = F.softmax(energy, dim=-1)                         # (B, L_q, L_k)

        # --- weighted sum of values ---
        v_proj  = self.W_v(kv)                                      # (B, L_k, q_dim)
        context = torch.bmm(alpha, v_proj)                          # (B, L_q, q_dim)

        return self.out_proj(context), alpha   # return weights for optional inspection


class Encoder_BahdanauAttention(nn.Module):
    """
    Patch encoder with Bahdanau cross-attention.

    Each patch is encoded locally; then each spatial token in the
    patch latent attends (additively) over the tokens of that patch's
    own embedding vector y_g.
    """
    def __init__(self, N: int, M: int, K: int, attn_dim: int | None = None, **kwargs):
        super().__init__(**kwargs)

        # [] review how far the cnn will go down
        self.local = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M, stride=1, kernel_size=3),
        )

        self.attn = BahdanauCrossAttention(q_dim=M, kv_dim=K, attn_dim=attn_dim)
        self.ln_q  = nn.LayerNorm(M)
        self.ln_kv = nn.LayerNorm(K)
        self.ln_out = nn.LayerNorm(M)

    def forward(
        self,
        x_p:  torch.Tensor,  # (B, P, C, Hp, Wp)  — patchified image
        y_g:  torch.Tensor,  # (B*P, K, h_g, w_g) — per-patch embedding
    ) -> torch.Tensor:       # (B*P, M, h, w)

        B, P, C, Hp, Wp = x_p.shape
        x_flat = x_p.reshape(B * P, C, Hp, Wp)

        # local latent  (B*P, M, h, w)
        y_local = self.local(x_flat)
        BP, M, h, w = y_local.shape

        # query tokens: spatial positions of the local latent
        q = y_local.flatten(2).transpose(1, 2)          # (BP, h*w, M)

        # key/value tokens: spatial positions of the patch embedding
        kv = y_g.flatten(2).transpose(1, 2).contiguous()  # (BP, h_g*w_g, K)

        # normalise before attention
        q  = self.ln_q(q)
        kv = self.ln_kv(kv)

        context, _ = self.attn(q, kv)   # (BP, h*w, M)

        # residual connection (pre-norm style)
        fused = self.ln_out(q + context)   # (BP, h*w, M)

        # reshape back to feature map
        y = fused.transpose(1, 2).reshape(BP, M, h, w)
        return y


class Decoder_BahdanauAttention(nn.Module):
    """
    Patch decoder with Bahdanau cross-attention.

    The quantised latent y_hat attends over the per-patch embedding
    before the deconv synthesis stack.
    """
    def __init__(self, N: int, M: int, K: int, attn_dim: int | None = None):
        super().__init__()
        self.M = M

        self.attn   = BahdanauCrossAttention(q_dim=M, kv_dim=K, attn_dim=attn_dim)
        self.ln_q   = nn.LayerNorm(M)
        self.ln_kv  = nn.LayerNorm(K)
        self.ln_out = nn.LayerNorm(M)

        self.ffn = nn.Sequential(
            nn.Linear(M, 4 * M),
            nn.GELU(),
            nn.Linear(4 * M, M),
        )
        self.ln_ffn = nn.LayerNorm(M)

        self.decode = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3, stride=1, kernel_size=3),
        )

    def forward(
        self,
        y_hat: torch.Tensor,  # (BP, M, h, w)
        y_g:   torch.Tensor,  # (BP, K, h_g, w_g)
    ) -> torch.Tensor:        # (BP, 3, Hp, Wp)

        BP, M, h, w = y_hat.shape

        q  = y_hat.flatten(2).transpose(1, 2)             # (BP, h*w, M)
        kv = y_g.flatten(2).transpose(1, 2).contiguous()  # (BP, h_g*w_g, K)

        q  = self.ln_q(q)
        kv = self.ln_kv(kv)

        context, _ = self.attn(q, kv)      # (BP, h*w, M)
        fused = self.ln_out(q + context)   # residual

        # optional FFN block (transformer-style)
        # fused = self.ln_ffn(fused + self.ffn(fused))

        y_fused = fused.transpose(1, 2).reshape(BP, M, h, w)
        return self.decode(y_fused)