# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model

from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from compressai.models.utils import conv, deconv


# [] ask toby about this issue, if running from train.py it is fine, but if running parameter_count.py then 
# i need to add src. beforehand, i understand why but how can i just avoid this issue and have safe consistent imports
# [] same thing  for the model after
from src.new_utils import patchify, embed_image, unpatchify, DownsampleCNN, LowResMask
from src.new_transforms import Encoder_CrossAttention, Decoder_CrossAttention

import gzip
import io

__all__ = [
    "CompressionModel",
    # "FactorizedPrior",
    # "FactorizedPriorReLU",
    "ScaleHyperprior",
    # "MeanScaleHyperprior",
    # "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]



#####
# first experiment model 
#####

@register_model("bmshj2018-hyperprior-crossattention")
class ScaleHyperpriorCrossAttention(CompressionModel):

    """
    Experimental scale hyperprior with cross attention
    """

    # [] the fact that the model takes in the embedding_model args is not clean, you can pass something else there and it won't be read, bad design, fix it
    def __init__(self, N, M, K, embedding_model, embedding_type, patch_size:int=16,  **kwargs):
        super().__init__(**kwargs)

        # fixing K size depending on embedding type
        # if we use a learned method,  we can determine K, but if not, it is not possible to determine K. 
        # if it is learned then just leave K as whatever was passed
        
        if embedding_type=="avgpool":
            embedding_model = LowResMask()
            K=3
        
        elif embedding_type == "downsample_cnn":
            embedding_model= DownsampleCNN(N,K) # K is not changed and is left as what is passed to original model

        self.entropy_bottleneck = EntropyBottleneck(N)  
        self.gaussian_conditional = GaussianConditional(None)
        self.y_ent_bot = EntropyBottleneck(N)

        self.g_a = Encoder_CrossAttention(N,M,K)
        self.g_s = Decoder_CrossAttention(N,M,K)

        self.embedding_model = embedding_model
        
        # self.embedding_model.eval()
        # for p in self.embedding_model.parameters():
        #     p.requires_grad = False

        # [] need to pass patch_size to encoder and decoder when initing them here
        # [] decode and encoder takes patched up image representations
        
        self.patch_size = patch_size
        self.N = int(N)
        self.M = int(M)
        self.K = int(K)

        
        # [] just rescale

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


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):


        y_g = self.embedding_model(x)
        # print("o_g y_g shape", y_g.shape)
        
        x_p = patchify(x, patch_size=self.patch_size)

        # print(f"x: {x.shape}\ty_g: {y_g.shape}\tx_p: {x_p.shape}")
        B, C, H, W = x.shape
        B, P, C, Hp, Wp = x_p.shape
        Gh = H // Hp
        Gw = W // Wp
        assert Gh * Gw == P, "patchify must tile the image regularly for this unpatchify"
        _ , _, h,w=y_g.shape
        # flattening patch dimension into vector dimension for x_p and y_g

        # [] is the flattening here necessary

        x_flat = x_p.reshape(B * P, C, Hp, Wp)
        # VERY IMPORTANT NOTE:
        # here the unsqueeze adds 1 dimension
        # then expand repeats the following dimensions P times in that dimension
        # meaning here, we have a single image repeated P times, for every single patch, 256 here
        # right now we are providing the full image as a global vector, but when we use bahdanau attention
        # this will not be correct, we cannot have a single patch (image) repeated P patches, but rather actual P patches
        
        y_g_flat = (
            y_g
            .unsqueeze(1)          # (B, 1, K)
            .expand(B, P, self.K, h,w)       # (B, P, K)
            .reshape(B * P, self.K, h,w)     # (B*P, K)
        )

        # y_g_transformed = self.conv_global_y(y_g_flat)

        # print(f"Pre encoder:{y_g_tansformed.shape}")
        
        y = self.g_a(x_p, y_g_flat)

        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)

        # print("shape z", z.shape)
        # print("shape z_hat", z_hat.shape)
        # print("shape y", y.shape)
        # print("scales shape", scales_hat.shape)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        
        y_g_hat, y_g_likelihoods = self.y_ent_bot(y_g_flat)

        # print(f"Pre decoder; {y_g_tansformed.shape}")
        x_hat_p_flat = self.g_s(y_hat, y_g_hat) 

        # If g_s outputs exactly patch size, this works directly.
        x_hat_p = x_hat_p_flat.reshape(B, P, 3, x_hat_p_flat.shape[-2], x_hat_p_flat.shape[-1])

        # ---- 10) Unpatchify to full image ----
        x_hat = unpatchify(x_hat_p, (Gh, Gw), )

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "y_g":y_g_likelihoods},
        }



# ==>> [] compress and decompress functions must be rewritten to account for changes in model
        # now you need to pass per image y_g_strings, y_strings, and x_strings 

    def compress(self, x):
        """
        The whole batch dimension thing is off and should be handeled differently
        shaping needs to be done better
        """
        # [] if all is well, the dimension of y_g of this function need fixing to be like the main model forward()

        x_p = patchify(x)
        B, C, H, W = x.shape
        B, P, C, Hp, Wp = x_p.shape
        Gh = H // Hp
        Gw = W // Wp

        y_g = self.embedding_model(x)
        _ , _, h,w= y_g.shape
        print("y_g shape", y_g.shape)

        # []  is the flattening also necesary  here  
        y_g_flat = (
            y_g
            .unsqueeze(1)          # (B, 1, K)
            .expand(B, P, self.K, h,w)       # (B, P, K)
            .reshape(B * P, self.K, h,w)     # (B*P, K)
        )

        # y_g_transformed = self.conv_global_y(y_g_flat)
        y = self.g_a(x_p, y_g_flat)

        z = self.h_a(torch.abs(y))

        ##### This lower part needs to be reviewed by someone else

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]) # [] what is this  extra thing with size and what does it do exactly??

        y_g_strings = self.y_ent_bot.compress(y_g_flat)

        # indexes just has the same shape as scales_hat
        scales_hat = self.h_s(z_hat)
        print("scales_hat shape", scales_hat.shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        #####
        print("y:", y.shape)
        print("z:", z.shape)
        print("z_hat:", z_hat.shape)
        print("scales_hat:", scales_hat.shape)
        print("indexes:", indexes.shape)
        # error below
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {"strings": [y_strings, z_strings, y_g_strings], "shape": [z.size()[-2:], y_g_flat.size()[-2:], [B,P,Gh,Gw]]}

    def decompress(self, strings, shape):

        # [] use this assert properly once things are more clear
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        y_g_hat = self.y_ent_bot.decompress(strings[2], shape[1])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        
        x_hat_p_flat = self.g_s(y_hat, y_g_hat).clamp_(0, 1) 
        x_hat_p = x_hat_p_flat.reshape(shape[2][0], shape[2][1], 3, x_hat_p_flat.shape[-2], x_hat_p_flat.shape[-1])

        # ---- 10) Unpatchify to full image ----
        x_hat = unpatchify(x_hat_p, (shape[2][2], shape[2][3]))
        
        return {"x_hat": x_hat}
