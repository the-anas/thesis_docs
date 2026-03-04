import torch
import torch.nn as nn
from compressai.models.utils import conv, deconv
from compressai.layers import GDN

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
            conv(N, M),  # -> (B*P, M, h', w')
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


        # [] maybe it is worth later to check if adding an mlp layer has any positive effect
        # self.ffn = nn.Sequential(
        #     nn.LayerNorm(M),
        #     nn.Linear(M, 4 * M),
        #     nn.GELU(),
        #     nn.Linear(4 * M, M),
        # )

    def forward(self, x_p, y_g):

        # [] YOU NEED BETTER SHAPING APPROACHES AND TO VERIFY THIS
        
        B, P, C, Hp, Wp = x_p.shape


        x_flat = x_p.reshape(B * P, C, Hp, Wp)
        print("x_flat shape", x_flat.shape)

        # ---- Local features (B*P, M, h', w') ----
        y_local = self.local(x_flat)
        BP, M, h, w = y_local.shape  # BP == B*P
        print("shape of y_locoal", y_local.shape)    

        print("x_g shape", x_p.shape)
        _,K,_,_ = y_g.shape
        print("y_g shape", y_g.shape)

        # ---- Turn feature map into query tokens: (B*P, L, M) where L=h*w ----
        # tokens correspond to spatial locations inside the patch latent
        q = y_local.flatten(2).transpose(1, 2)  # (BP, L, M)

        print("shape of q", q.shape)
        # ---- Global context as 1 token per patch: (B*P, 1, K) ----
        kv = y_g.reshape(B * P, 1, K) # attempted fix for avg_pool K*16*16

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
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
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

        kv = y_g.reshape(BP, 1, self.K)  # (BP, 1, K)

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
