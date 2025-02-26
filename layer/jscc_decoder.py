import math
import torch.nn as nn
import torch

from timm.models.layers import trunc_normal_
import numpy as np
from layer.jscc_encoder import JSCCEncoder
from layer.vmamba import  VSSLayer_up
class RateAdaptionDecoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.weight = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.rate_num)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.weight_bias, std=.02)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, max(self.rate_choice), self.C)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, self.C)
        # print(w.dtype)
        # print(b.dtype)
        # print(x_BLC.dtype)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b  # BLN
        out = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out

class JSCCDecoder(nn.Module):
    def __init__(self,
                 embed_dim=256, depths=[1, 1, 1],
                 norm_layer=nn.LayerNorm,rate_choice=[16, 160, 256],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 d_state=16,
    ):
        super(JSCCDecoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = VSSLayer_up(
                dim_in=embed_dim,
                dim_out=embed_dim,
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=None,
                use_checkpoint=False,
            )
            self.layers.append(layer)
        self.embed_dim = embed_dim
        self.rate_adaption = RateAdaptionDecoder(embed_dim, rate_choice)
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x = self.rate_adaption(x, indexes)
        x_BLC = x.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, self.embed_dim)
        x_BLC = x_BLC + rate_token
        x_BHWC=x_BLC.reshape(B,H,W,self.embed_dim)
        for layer in self.layers:
            x_BHWC = layer(x_BHWC.contiguous())
        x_BCHW = x_BHWC.permute(0, 3, 1, 2)
        return x_BCHW
    
    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * 2, W * 2)

multiple_rate = [16,32,48,64,80,96,102,118, 134, 160, 186, 192, 208, 224,240, 256]
fd_kwargs = dict(
        embed_dim=256, depths=[1,1,1],norm_layer=nn.LayerNorm, rate_choice=multiple_rate,
        drop_rate=0.,attn_drop_rate=0., drop_path_rate=0.1,d_state=16,
    )

if __name__=="__main__":
    fe=JSCCEncoder(**fd_kwargs)
    fe=fe.cuda()
    x=torch.randn(1,256,16,16).cuda()
    x,_,y=fe(x,x,0.2)
    fd=JSCCDecoder(**fd_kwargs)
    fd=fd.cuda()
    out=fd(x,y)
    print(out.shape)