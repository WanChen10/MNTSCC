import math
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from layer.vmamba import Mlp
import numpy as np
from layer.vmamba import VSSLayer
class RateAdaptionEncoder(nn.Module):
    def __init__(self, channel_num, rate_choice,img_size, mode='CHW'):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.H, self.W = (channel_num, img_size, img_size)#输入张量的通道数，以及高度和宽度
        self.rate_num = len(rate_choice)#可选速率数量
        self.rate_choice = rate_choice
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        print("CONFIG RATE", self.rate_choice_tensor)#将可选速率设置为一个张量
        self.weight = nn.Parameter(torch.zeros(self.rate_num, self.C, max(self.rate_choice)))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.w, std=.02)
        mask = torch.arange(0, max(self.rate_choice)).repeat(self.H * self.W, 1)
        self.register_buffer("mask", mask)

    def forward(self, x, indexes):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        # if H != self.H or W != self.W:
        #     self.update_resolution(H, W, x.get_device())
        #通过indexes从权重中选择对应速率
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, self.C, -1)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, -1)
        #mask用于存储速率约束的张量
        mask = self.mask.repeat(B, 1, 1)
        
        rate_constraint = self.rate_choice_tensor[indexes].reshape(B, H * W, 1).repeat(1, 1, max(self.rate_choice))
        mask_new = torch.zeros_like(mask)
        mask_new[mask < rate_constraint] = 1
        mask_new[mask >= rate_constraint] = 0
        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b) * mask_new
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask_new.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_masked, mask_BCHW

    # def update_resolution(self, H, W, device):
    #     self.H = H
    #     self.W = W
    #     self.num_patches = H * W
    #     self.mask = torch.arange(0, max(self.rate_choice)).repeat(self.num_patches, 1)
    #     self.mask = self.mask.to(device)

class JSCCEncoder(nn.Module):
    def __init__(self,
                 embed_dim=256, depths=[1, 1, 1],
                 norm_layer=nn.LayerNorm,
                 rate_choice=[16, 160, 256],
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 d_state=16,
                 img_size=16
                ):
        super(JSCCEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(depths)):
            layer = VSSLayer(
                dim_in=embed_dim,
                dim_out=embed_dim,
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
            )
            self.layers.append(layer)
        self.rate_adaption = RateAdaptionEncoder(embed_dim, rate_choice,img_size)#定义速率模块
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)
        self.refine = Mlp(embed_dim * 2, embed_dim * 8, embed_dim)
        self.norm = norm_layer(embed_dim)

    def forward(self, x, px, eta):
        """
        JSCCEncoder encodes latent representations to variable length channel-input vector.

        Arguments:
        x: Latent representation (patch embeddings), shape of BxCxHxW, also viewed as Bx(HxW)xC.
        px: Estimated probability of x, shape of BxCxHxW, also viewed as Bx(HxW)xC.
        eta: Scaling factor from entropy to channel bandwidth cost.

        Returns:
        s_masked: Channel-input vector.
        indexes: The length of each patch embedding, shape of BxHxW.
        mask: Binary mask, shape of BxCxHxW.
        """

        B, C, H, W = x.size()
        hx = torch.clamp_min(-torch.log(px) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        x_BLC = x.flatten(2).permute(0, 2, 1)#将bchw——》》b (h*w) c
        px_BLC = px.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)  # B*H*W
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, C)
        x_BLC = x_BLC + rate_token
        x_BHWC=x_BLC.reshape(B,H,W,C)
        # for layer in self.layers:
        #     x_BLC = layer(x_BLC.contiguous())
        # x_BLC = self.norm(x_BLC)
        # x_BCHW = x_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
        for layer in self.layers:
            x_BHWC=layer(x_BHWC.contiguous())
        x_BCHW=x_BHWC.permute(0,3,1,2)
        s_masked, mask = self.rate_adaption(x_BCHW, indexes)
        return s_masked, mask, indexes

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * 2, W * 2)




multiple_rate = [16,32,48,64,80,96,102,118, 134, 160, 186, 192, 208, 224,240, 256]
fe_kwargs = dict(
        embed_dim=256, depths=[1,1,1],norm_layer=nn.LayerNorm, rate_choice=multiple_rate,
        drop_rate=0.,attn_drop_rate=0., drop_path_rate=0.1,d_state=16,
    )

if __name__=="__main__":

    fe=JSCCEncoder(**fe_kwargs)
    fe=fe.cuda()
    x=torch.randn(1,256,16,16).cuda()
    _,_,out=fe(x,x,0.2)
    print(out.shape)
