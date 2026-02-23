#bash -lc cat > /mnt/data/scale_hyperprior_memstream.py <<'PY'
import math
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.registry import register_model

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv, deconv, GDN

# NOTE:
# This code assumes the same CompressAI-style utilities as your original class:
# - CompressionModel
# - EntropyBottleneck
# - GaussianConditional
# - conv, deconv, GDN
# - register_model
# If you're in the CompressAI repo, these are typically available under:
#   from compressai.models import CompressionModel
#   from compressai.entropy_models import EntropyBottleneck, GaussianConditional
#   from compressai.layers import conv, deconv, GDN
#   from compressai.registry import register_model


#####
# patch related funcs
####


# helper func for calculating padding size and patch related resizing
def _ceil_to_multiple(x: int, m: int) -> int:
    """Small helper: smallest multiple of m >= x."""
    return ((x + m - 1) // m) * m


def _pad_to_multiple_of_patch(x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """EDIT: Added padding helper for streaming patches.

    We want to handle arbitrary H, W at inference.
    If H or W is not divisible by patch_size, we pad (right/bottom) and remember original size.

    Returns:
        x_pad: padded tensor
        orig_hw: (H_orig, W_orig)
    """
    assert x.dim() == 4, "Expected BCHW"
    _, _, H, W = x.shape
    H2 = _ceil_to_multiple(H, patch_size)
    W2 = _ceil_to_multiple(W, patch_size)
    pad_h = H2 - H
    pad_w = W2 - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)

    # Pad format for F.pad is (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x_pad, (H, W)


def _unpad_to_orig(x_hat: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    """Remove the right/bottom padding added by _pad_to_multiple_of_patch."""
    H, W = orig_hw
    return x_hat[..., :H, :W]

# single output elem in tuple contains tensor, height, width 
def _extract_patches(x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """EDIT: Added patch extraction for streaming.

    Converts BCHW image to (B, L, C, P, P) patches using non-overlapping tiles.

    Returns:
        patches: (B, L, C, P, P)
        nph: number of patches along height
        npw: number of patches along width
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Input must be padded to patch multiple"

    nph = H // patch_size
    npw = W // patch_size

    # Unfold into patches: (B, C*P*P, L)
    patches_flat = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    L = patches_flat.shape[-1]

    # Reshape to (B, L, C, P, P)
    patches = patches_flat.transpose(1, 2).contiguous().view(B, L, C, patch_size, patch_size)
    return patches, nph, npw


def _fold_patches(patches: torch.Tensor, patch_size: int, nph: int, npw: int) -> torch.Tensor:
    """EDIT: Added patch folding for streaming.

    Converts patches (B, L, C, P, P) back to an image (B, C, H, W).
    Assumes non-overlapping.
    """
    B, L, C, P, P2 = patches.shape
    assert P == patch_size and P2 == patch_size
    assert L == nph * npw

    H = nph * patch_size
    W = npw * patch_size

    patches_flat = patches.view(B, L, C * patch_size * patch_size).transpose(1, 2).contiguous()
    x = F.fold(patches_flat, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return x



####
#model
####


@register_model("bmshj2018-hyperprior-memstream")
class ScaleHyperpriorMemoryStream(nn.Module):
    """EDIT: A streaming/patchwise enhancement of bmshj2018 Scale Hyperprior.

    What changed vs the original model you posted:

    1) EDIT (streaming): We process the image as a sequence of patches (one patch per timestep).
    2) EDIT (Option B memory): We maintain a global *memory canvas* M that is updated per patch.
       - M is NOT pixels. It's a low-res *feature map* storing learned global features.
    3) EDIT (entropy model conditioning): We condition the scale prediction for y on memory context.
       - scales_hat = h_s(z_hat) * exp(delta_log_scale(context))
    4) EDIT (residualization): We predict a global component from memory and subtract it:
       - r = y - y_global(context)
       - We entropy-code r instead of y, so patch latents carry less "global" information.
    5) EDIT (codec validity): Memory updates use only decoded-safe quantities (z_hat by default),
       so the decoder can reproduce M exactly.

    This class is written to be drop-in compatible with CompressAI-style usage *conceptually*.
    If you're inside CompressAI, you should inherit from CompressionModel and reuse its helpers.
    Here we keep it as nn.Module for self-contained clarity; adapt inheritance as needed.

    Args:
        N: hyperprior channels (same as original)
        M: main latent channels (same as original)
        patch_size: size of streaming patches
        mem_channels: channels of memory canvas M
        pos_dim: dimensionality of position embedding
        use_global_pool: whether to use global pooled memory vector in context
        update_from: "z" (default) or "y" - which decoded-safe signal to write into memory
    """

    def __init__(
        self,
        N: int,
        M: int,
        patch_size: int = 256, # this most likely means 16*16 patches
        mem_channels: Optional[int] = None,
        pos_dim: int = 64, # [] is this the dimension of the positional embedding??
        use_global_pool: bool = True,
        update_from: str = "z",
        **kwargs: Any,
    ):
        super().__init__()

        # ------------------------
        # Original bmshj2018 parts
        # ------------------------
        

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        # ------------------------
        # EDIT: Streaming settings
        # ------------------------
        self.patch_size = int(patch_size)
        self.use_global_pool = bool(use_global_pool) 
        self.update_from = str(update_from)
        assert self.update_from in {"z", "y"}, "update_from must be 'z' or 'y'"
        # [] what does it mean to update from z?
        # what tensor exaclty is being updated and how is the update made?

        # ------------------------
        # EDIT: Memory canvas setup
        # ------------------------
        # Default: memory channels = N (keeps things simple and cheap)
        # => so memory channels == hyperprior channels 
        # [] maybe you can experiment with this
        self.mem_channels = int(mem_channels) if mem_channels is not None else self.N


        # [] is this below suppsoed to be a learnable positional embedding??
        # [] if so how and when is it added to the patch embeddings??
        # [X] why is it taking a fixed number 2 of in_features?
        # => takes in the normalized (row, col), that is why it is 2
        
        # EDIT: Position embedding from normalized (row, col) in [0,1].
        # This (normalization to [0,1]) avoids needing a fixed max grid size.
        self.pos_dim = int(pos_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )

        # [] is this a concatination? [] why is global memory optional?
        # [] is this below taking positional embeddings or just taking an empty tensor of their size??
        # [] difference between ctx_dim and ctx_in

        # EDIT: Memory read combiner.
        # Context vector c_t = [local_memory, global_memory(optional), pos_emb]
        ctx_in = self.mem_channels + (self.mem_channels if self.use_global_pool else 0) + self.pos_dim
        self.ctx_dim = ctx_in


        # [] what is this scale_cond? does a single linear operation result in conditionning?
        # [] how is conditionning defined in earlier parts of the code 
        # EDIT: (A) Conditioning scales_hat using context.
        # We predict a per-channel delta_log_scale in R^M and apply scales *= exp(delta).
        self.scale_cond = nn.Sequential(
            nn.Linear(self.ctx_dim, self.M),
        )

        # [] what is the difference between this and the operation above
        # EDIT: (B) Predict global component of y from context.
        # Minimal, stable choice: channel-wise bias (broadcasted over spatial dims).
        self.y_global_pred = nn.Sequential(
            nn.Linear(self.ctx_dim, self.M),
        )


        # [] what are all these different tensors defined in __init__
        # and how do they relate to memory??


        # [] what is this dilemma of updating between y_hat and z_hat? where is it coming from??
        # [] how come the pooling operation is not taking any arguments??
        
        
        # EDIT: Memory write projection.
        # We need to turn decoded-safe tensors (z_hat or y_hat) into a vector in R^{mem_channels}.
        # - If update_from == "z": input has N channels (z_hat)
        # - If update_from == "y": input has M channels (y_hat)
        in_ch = self.N if self.update_from == "z" else self.M
        self.write_proj = nn.Sequential(
            # [] i think this takes a latent representation of a patch
            # and then projects it into its portion on the memory canvas, 
            # is this correct??
            # [] so we are only taking the largest value across 
            # the entire latent representation of the patch??
            # [] what makes it adapative
            nn.AdaptiveAvgPool2d(1),  # global average pool over patch-latent spatial dims
            nn.Flatten(1),            # (B, in_ch)
            nn.Linear(in_ch, self.mem_channels),
        )

        
        # EDIT: Learned gated write (optional but helpful).
        # Gate decides how much to keep of previous memory cell.
        self.write_gate = nn.Sequential(
            nn.Linear(self.mem_channels + self.mem_channels + self.pos_dim, self.mem_channels),
            nn.Sigmoid(),
        )

    # ------------------------
    # EDIT: Convenience: build memory tensor given grid size
    # ------------------------
    def _init_memory(self, B: int, nph: int, npw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create zero-initialized memory canvas M_0.

        Shape: (B, C_m, nph, npw) if we use one cell per patch.
        """
        return torch.zeros((B, self.mem_channels, nph, npw), device=device, dtype=dtype)

    # ------------------------
    # EDIT: Read context c_t from memory and position
    # ------------------------

    # [] i thought we already had positional embeddings defined in __init__?
    # [] what does each one do?
    def _pos_embed(self, r: int, c: int, nph: int, npw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Position embedding for patch at (r,c) normalized to [0,1]."""
        # Normalize to [0,1]; add small eps to avoid division by zero when nph/npw==1
        rr = 0.0 if nph <= 1 else r / float(nph - 1)
        cc = 0.0 if npw <= 1 else c / float(npw - 1)
        xy = torch.tensor([[rr, cc]], device=device, dtype=dtype)  # (1,2)
        return self.pos_mlp(xy)  # (1,pos_dim)

    def _read_context(
        self,
        M_canvas: torch.Tensor,
        r: int,
        c: int,
        nph: int,
        npw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build context vector c_t.

        M_canvas: (B, C_m, nph, npw)
        Returns: c_t of shape (B, ctx_dim)
        """
        # Local read: the memory cell corresponding to this patch.
        # (B, C_m)
        c_local = M_canvas[:, :, r, c]

        # Optional global read: average over all memory cells.
        if self.use_global_pool:
            # (B, C_m)
            c_global = M_canvas.mean(dim=(2, 3))
            mem_parts = [c_local, c_global]
        else:
            mem_parts = [c_local]

        # Position embedding: (1,pos_dim) -> broadcast to (B,pos_dim)
        pos = self._pos_embed(r, c, nph, npw, device=device, dtype=dtype)
        pos = pos.expand(M_canvas.size(0), -1)

        # Concatenate into a single context vector.
        c_t = torch.cat(mem_parts + [pos], dim=1)
        return c_t

    # ------------------------
    # EDIT: Write/update memory
    # ------------------------
    def _write_memory(
        self,
        M_canvas: torch.Tensor,
        write_src: torch.Tensor,
        r: int,
        c: int,
        nph: int,
        npw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Update memory cell (r,c) using decoded-safe write_src.

        write_src is typically z_hat (preferred) or y_hat.

        - We compute u_t = write_proj(write_src): (B, C_m)
        - We gate-update the current cell:
            M_new = g * M_old + (1-g) * u_t
        """
        B = M_canvas.size(0)

        # Convert decoded-safe tensor -> write vector.
        u_t = self.write_proj(write_src)  # (B, C_m)

        # Current cell.
        m_old = M_canvas[:, :, r, c]  # (B, C_m)

        # Gate depends on old cell, new proposal, and position.
        pos = self._pos_embed(r, c, nph, npw, device=device, dtype=dtype).expand(B, -1)
        gate_in = torch.cat([m_old, u_t, pos], dim=1)  # (B, 2*C_m + pos_dim)
        g = self.write_gate(gate_in)  # (B, C_m) in (0,1)

        m_new = g * m_old + (1.0 - g) * u_t

        # Write back (copy to avoid in-place autograd issues in training).
        M_next = M_canvas.clone()
        M_next[:, :, r, c] = m_new
        return M_next

    # ------------------------
    # EDIT: Context-conditioned scales
    # [] what is a scale or what are scales?? how do they relate to memory?
    # [] where is the conditionning and what does h_s have to do with this function?
    # ------------------------
    def _condition_scales(self, scales_hat: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """Condition the predicted scales on context.

        scales_hat from h_s is already non-negative due to final ReLU.
        We apply multiplicative conditioning that preserves positivity:
            scales_hat *= exp(delta_log_scale)
        """
        delta = self.scale_cond(c_t)  # (B, M)
        delta = delta.view(delta.size(0), delta.size(1), 1, 1)  # (B, M, 1, 1)

        # Keep positivity by multiplication.
        scales = scales_hat * torch.exp(delta)

        # Numerical safety: GaussianConditional expects strictly positive scales.
        scales = scales.clamp_min(1e-6)
        return scales

    # ------------------------
    # EDIT: Global component predictor for y
    # [] why are we predicting y using global vector? i dont understand this?
    # this is not just done when decoding 
    # ------------------------
    def _predict_y_global(self, y: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """Predict the part of y that should be "explained" by global memory.

        Minimal stable choice: per-channel bias (broadcast over spatial).
        Output shape matches y: (B, M, Hy, Wy)
        """
        mu = self.y_global_pred(c_t)  # (B, M)
        mu = mu.view(mu.size(0), mu.size(1), 1, 1)
        return mu.expand_as(y)

    # ------------------------
    # EDIT: Forward (training) processes full image as patches
    # [] from func doc, is there no memory being used here or what exactly?
    # ------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass for training.

        Returns the same kind of dict as the original:
          {"x_hat": ..., "likelihoods": {"y": ..., "z": ...}}

        BUT now likelihoods are aggregated over patches.
        """
        # [] why are patches being exatracted in the forward function?
        # are they gonna rerun with every time step?
        # shouldn't this be done with some sort of dataloader earlier on? 

        # EDIT: Pad to multiple of patch size.
        x_pad, orig_hw = _pad_to_multiple_of_patch(x, self.patch_size)

        patches, nph, npw = _extract_patches(x_pad, self.patch_size)
        B, L, C, P, _ = patches.shape

        device = x.device
        dtype = x.dtype

        # EDIT: Initialize memory M_0.
        M_canvas = self._init_memory(B, nph, npw, device=device, dtype=dtype)

        # [] again, if forward is the training function, 
        # how come this is initualized here? is it not going to be re init every time?

        # We'll store reconstructed patches.
        xhat_patches: List[torch.Tensor] = []

        # We'll store likelihood tensors for y (actually residual r) and z.
        # Each likelihood is shape like the latent tensor.
        y_likes: List[torch.Tensor] = []
        z_likes: List[torch.Tensor] = []

        # [] a single run is reading the enttire image?? patch per patch??
        # [] i think training should be structured more like the original model?
        # [] how is that structured and how can i replicate it?

        # Process patches in raster order.
        for t in range(L):
            r = t // npw
            c = t % npw

            x_t = patches[:, t, :, :, :]  # (B, 3, P, P)

            # 1) READ: context from memory for this patch.
            c_t = self._read_context(M_canvas, r, c, nph, npw, device=device, dtype=dtype)  # (B, ctx_dim)

            # 2) Encode patch -> y.
            y = self.g_a(x_t)

            # 3) Hyper-encode -> z.
            z = self.h_a(torch.abs(y))

            # 4) Entropy bottleneck (training version returns z_hat + likelihoods).
            z_hat, z_likelihoods = self.entropy_bottleneck(z)

            # 5) Predict scales from z_hat.
            scales_hat = self.h_s(z_hat)

            # 6) EDIT: Condition scales on global memory context.
            scales_hat = self._condition_scales(scales_hat, c_t)

            # 7) EDIT: Predict y_global from memory and subtract to get residual r.
            y_global = self._predict_y_global(y, c_t)
            r_latent = y - y_global

            # 8) Code residual with GaussianConditional (training version returns r_hat + likelihoods).
            r_hat, y_likelihoods = self.gaussian_conditional(r_latent, scales_hat)

            # 9) Reconstruct y_hat and then x_hat patch.
            y_hat = r_hat + y_global
            x_hat_t = self.g_s(y_hat)

            # 10) EDIT: Update memory using decoded-safe signal.
            # By default, update from z_hat (available on decoder immediately after z decode).
            write_src = z_hat if self.update_from == "z" else y_hat
            M_canvas = self._write_memory(M_canvas, write_src, r, c, nph, npw, device=device, dtype=dtype)

            xhat_patches.append(x_hat_t)
            y_likes.append(y_likelihoods)
            z_likes.append(z_likelihoods)

        # Stack patches back to image.
        xhat_stack = torch.stack(xhat_patches, dim=1)  # (B, L, 3, P, P)
        x_hat_pad = _fold_patches(xhat_stack, self.patch_size, nph, npw)
        x_hat = _unpad_to_orig(x_hat_pad, orig_hw)

        # Aggregate likelihoods by concatenation over patches.
        # We keep them as lists in case your trainer sums -log(likelihood) anyway.
        # If you prefer a single tensor, you can torch.cat along a new dimension.
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likes,  # list of tensors, one per patch
                "z": z_likes,  # list of tensors, one per patch
            },
            # Extra metadata can help debugging.
            "_meta": {
                "orig_hw": orig_hw,
                "nph": nph,
                "npw": npw,
                "patch_size": self.patch_size,
            },
        }

    # [] compare above forward func to original forward func
    # [] also review differences between compress and decompress funcs
    # here and in original model 

    # ------------------------
    # EDIT: Streaming compress
    # ------------------------
    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compress an image by streaming patches.

        Returns a dict similar to original, but strings are now lists per patch.

        IMPORTANT:
        - This implementation assumes B==1 for simplicity (common for codecs).
          You can extend to B>1 by looping over batch and storing per-sample lists.
        """
        if x.size(0) != 1:
            raise ValueError("This reference compress() expects batch size 1 for simplicity.")

        x_pad, orig_hw = _pad_to_multiple_of_patch(x, self.patch_size)
        patches, nph, npw = _extract_patches(x_pad, self.patch_size)
        _, L, _, _, _ = patches.shape

        device = x.device
        dtype = x.dtype

        # Initialize memory.
        M_canvas = self._init_memory(1, nph, npw, device=device, dtype=dtype)

        # We'll store bitstrings per patch.
        y_strings: List[bytes] = []  # residual r strings
        z_strings: List[bytes] = []

        # We also need z spatial shape for entropy_bottleneck.decompress.
        z_spatial: Optional[Tuple[int, int]] = None

        for t in range(L):
            r = t // npw
            c = t % npw

            x_t = patches[:, t, :, :, :]

            # Read context from memory (uses previous patches only).
            c_t = self._read_context(M_canvas, r, c, nph, npw, device=device, dtype=dtype)

            # Encode patch -> y, z.
            y = self.g_a(x_t)
            z = self.h_a(torch.abs(y))

            if z_spatial is None:
                z_spatial = (int(z.size(-2)), int(z.size(-1)))

            # ---- Encode z first (same ordering as original hyperprior) ----
            z_str_list = self.entropy_bottleneck.compress(z)  # list length B
            z_str = z_str_list[0]
            z_strings.append(z_str)

            # Decoder will reconstruct z_hat from z_str; encoder does it too to stay consistent.
            z_hat = self.entropy_bottleneck.decompress([z_str], z_spatial)

            # Predict scales and condition them on context.
            scales_hat = self.h_s(z_hat)
            scales_hat = self._condition_scales(scales_hat, c_t)

            # Predict global component and residualize.
            y_global = self._predict_y_global(y, c_t)
            r_latent = y - y_global

            # Build indexes and compress residual.
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            r_str_list = self.gaussian_conditional.compress(r_latent, indexes)
            r_str = r_str_list[0]
            y_strings.append(r_str)

            # Update memory using decoded-safe quantities.
            # We must mirror the decoder; thus we use z_hat (or y_hat if selected).
            if self.update_from == "z":
                write_src = z_hat
            else:
                # If updating from y, we need y_hat; decode residual now.
                r_hat = self.gaussian_conditional.decompress([r_str], indexes, z_hat.dtype)
                y_hat = r_hat + y_global
                write_src = y_hat

            M_canvas = self._write_memory(M_canvas, write_src, r, c, nph, npw, device=device, dtype=dtype)

        assert z_spatial is not None

        return {
            # EDIT: store per-patch strings.
            "strings": [y_strings, z_strings],
            # EDIT: need metadata to reconstruct patch grid and z shape.
            "meta": {
                "shape_z": z_spatial,
                "orig_hw": orig_hw,
                "nph": nph,
                "npw": npw,
                "patch_size": self.patch_size,
                "update_from": self.update_from,
                "use_global_pool": self.use_global_pool,
                "mem_channels": self.mem_channels,
            },
        }

    # ------------------------
    # EDIT: Streaming decompress
    # ------------------------
    @torch.no_grad()
    def decompress(self, strings: List[List[bytes]], meta: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress a streamed bitstream.

        Args:
            strings: [y_strings, z_strings], each a list of length L
            meta: metadata from compress()

        Returns:
            {"x_hat": reconstructed image}
        """
        assert isinstance(strings, list) and len(strings) == 2
        y_strings, z_strings = strings
        assert len(y_strings) == len(z_strings), "y and z must have same number of patches"

        L = len(y_strings)

        shape_z = tuple(meta["shape_z"])  # (Hz, Wz)
        orig_hw = tuple(meta["orig_hw"])  # (H_orig, W_orig)
        nph = int(meta["nph"])
        npw = int(meta["npw"])
        patch_size = int(meta["patch_size"])

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Initialize memory.
        M_canvas = self._init_memory(1, nph, npw, device=device, dtype=dtype)

        xhat_patches: List[torch.Tensor] = []

        for t in range(L):
            r = t // npw
            c = t % npw

            # Read context from memory.
            c_t = self._read_context(M_canvas, r, c, nph, npw, device=device, dtype=dtype)

            # Decode z_hat for this patch.
            z_hat = self.entropy_bottleneck.decompress([z_strings[t]], shape_z)

            # Predict scales and condition on context.
            scales_hat = self.h_s(z_hat)
            scales_hat = self._condition_scales(scales_hat, c_t)

            # Build indexes.
            indexes = self.gaussian_conditional.build_indexes(scales_hat)

            # Predict y_global (decoder can do this from context alone).
            # BUT we need a tensor with the right spatial size. We don't know Hy,Wy yet.
            # Solution: decode residual r_hat first, then create y_global by expanding.
            r_hat = self.gaussian_conditional.decompress([y_strings[t]], indexes, z_hat.dtype)

            # Create y_global with correct shape by using r_hat as a shape reference.
            y_global = self._predict_y_global(r_hat, c_t)

            # Reconstruct y_hat and patch.
            y_hat = r_hat + y_global
            x_hat_t = self.g_s(y_hat)
            xhat_patches.append(x_hat_t)

            # Update memory.
            write_src = z_hat if self.update_from == "z" else y_hat
            M_canvas = self._write_memory(M_canvas, write_src, r, c, nph, npw, device=device, dtype=dtype)

        # Fold patches back.
        xhat_stack = torch.stack(xhat_patches, dim=1)  # (1, L, 3, P, P)
        x_hat_pad = _fold_patches(xhat_stack, patch_size, nph, npw)
        x_hat = _unpad_to_orig(x_hat_pad, orig_hw).clamp_(0, 1)

        return {"x_hat": x_hat}
        
#PY
#python -m py_compile /mnt/data/scale_hyperprior_memstream.py


