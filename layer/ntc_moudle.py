
import torch.nn as nn
import torch
import math
from sympy import print_rcode

try:
    from layer.vmamba import VSSLayer,VSSLayer_up,PatchMerging2D,PatchExpand2D,PatchEmbed2D
except ImportError:
    from layer.vmamba import VSSLayer,VSSLayer_up,PatchEmbed2D,PatchExpand2D,PatchMerging2D

class ga(nn.Module):
    #input 1,3,256,256 output 1，256，16，16
    def __init__(self, patch_size=2, in_chans=3, depths=[1, 1, 2, 4], embed_dim=[256,256,256,256] ,
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.embed=PatchEmbed2D(patch_size=patch_size, in_chans=in_chans,
                                embed_dim=embed_dim[0],
                                norm_layer=norm_layer if patch_norm else None)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim_in=embed_dim[i_layer-1] if i_layer>0 else embed_dim[i_layer],
                dim_out=embed_dim[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim[i_layer] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer!=0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        x=self.embed(x)
        for layer in self.layers:
            x = layer(x)
        y=x.permute(0,3,1,2)
        return y


class gs(nn.Module):
    # input 1,256，16，16  output:1,3,256,256
    def __init__(self,
                 out_chans=3,
                 depths = [4, 2, 1, 1],embed_dim=[256,256,256,256],
                 d_state = 16, drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm, use_checkpoint = False, **kwargs
                 ):
        super().__init__()
        self.num_layers = len(depths)
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))][::-1]
        self.final_conv = nn.ConvTranspose2d(in_channels=embed_dim[-1], out_channels=out_chans, kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim_in=embed_dim[i_layer],
                dim_out=embed_dim[i_layer+1] if i_layer+1<len(embed_dim) else embed_dim[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim[i_layer] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        y = x.permute(0, 3, 1, 2)
        y = self.final_conv(y)
        return y
class ha(nn.Module):
    #input 1，256，16，16  output 1,192,4,4
    def __init__(self, depths_ha=[2, 2], embed_dim=192, d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_ha))]
        self.num_layers = len(depths_ha)
        self.layers = nn.ModuleList()
        for i_layer in range(2): 
            layer = VSSLayer(
                dim_in=256 if i_layer==0 else embed_dim,
                dim_out=embed_dim,
                depth=depths_ha[i_layer],
                d_state=d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_ha[:i_layer]):sum(depths_ha[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
    def forward(self, x):
        x=x.permute(0,2,3,1)#bchw->>bhwc
        for layer in self.layers:
            x = layer(x)
        z=x.permute(0,3,1,2)#bhwc->>bchw
        return z#1,256,16,16 ->>1,448,4,4
class hs(nn.Module):
    #input 1,192，4，4  output 1,512,16,16
    def __init__(self,  depths_hs=[2, 2], 
                 embed_dim=[192,384,512], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths_hs)
        self.hs = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_hs))]
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim_in=embed_dim[i_layer],
                dim_out=embed_dim[i_layer+1],
                depth=depths_hs[i_layer],
                d_state=d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_hs[:i_layer]):sum(depths_hs[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D,
                use_checkpoint=use_checkpoint,
            )
            self.hs.append(layer)

    def forward(self, x):
        x=x.permute(0,2,3,1)
        for layer in self.hs:
            x = layer(x)
        x=x.permute(0,3,1,2)
        return x

class ga_cifar(nn.Module):
    #input 1,3,256,256 output 1，256，16，16
    def __init__(self, patch_size=2, in_chans=3, depths=[2, 4], embed_dim=[128,256], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.embed=PatchEmbed2D(patch_size=patch_size, 
                                in_chans=in_chans, 
                                embed_dim=embed_dim[0],
                                norm_layer=norm_layer if patch_norm else None)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim_in=embed_dim[i_layer],
                dim_out=256,
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer==0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        x=self.embed(x)
        for layer in self.layers:
            x = layer(x)
        y=x.permute(0,3,1,2)
        return y
class gs_cifar(nn.Module):
    #input 1,256，16，16  output:1,3,256,256
    def __init__(self,  depths=[4, 2], 
                embed_dim=256, d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):

        super().__init__()
        self.num_layers = len(depths)
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))][::-1]
        self.final_conv=nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=2,stride=2)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim_in=embed_dim if i_layer==0 else 128,
                dim_out=128 ,
                depth=depths[i_layer],
                d_state=math.ceil(embed_dim / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer ==0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        
    def forward(self, x):
        x=x.permute(0,2,3,1)
        for layer in self.layers:
            x = layer(x)
        y = x.permute(0, 3, 1, 2)
        y=self.final_conv(y)
        return y

ga_kwargs = dict(
        patch_size=2,in_chans=3,
        depths=[1,1,2, 4],embed_dim=[256,256,256, 256],
        d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False
    )
gs_kwargs = dict(
        out_chans=3,
        depths=[4,2,1,1],embed_dim=[256,256,256,256],
        d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False
    )

if __name__=="__main__":
        ga=ga(**ga_kwargs)
        ga=ga.cuda()
        x=torch.randn((1,3,256,256)).cuda()
        y=ga(x)
        gs=gs(**gs_kwargs)
        gs=gs.cuda()
        y=gs(y)
