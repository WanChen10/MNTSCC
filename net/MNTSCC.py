import torch
import math
import torch.nn as nn
from loss.distortion import Distortion
from compressai.ops import quantize_ste
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.layers import (
    AttentionBlock
)
from layer.vmamba import Mlp
from layer.ntc_moudle import gs,ga,ha,hs
from layer.vmamba import VSSBlock
from layer.jscc_encoder import JSCCEncoder
from layer.jscc_decoder import JSCCDecoder
from utils import BCHW2BLN, BLN2BCHW
from channel.A_channel import Channel

from compressai.models.utils import conv

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
class MambaAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim,inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = VSSBlock(
                # stage=stage,
                hidden_dim=inter_dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=16,
            )
        else:
            super().__init__(N=input_dim)
            self.non_local_block = VSSBlock(
                # stage=stage,
                hidden_dim=input_dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=16,
            )
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        # pdb.set_trace()
        x = self.in_conv(x)
        identity = x
        x_m=x.permute(0,2,3,1)
        z = self.non_local_block(x_m)
        z=z.permute(0,3,1,2)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out   
class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )
    def forward(self, snr):
        #根据m，将snr从(b,1)转为(b,m) 
        return self.fc(snr)
class snr_net(nn.Module):
    def __init__(self, embded_dim):
        super(snr_net, self).__init__()
        self.embded_dim = embded_dim
        self.hidden_dim = int(embded_dim*1.5)
        self.sm_list = nn.ModuleList()
        self.bm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embded_dim, self.hidden_dim))
        self.layer_num = 7
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                outdim = self.embded_dim
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,snr):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        device = x.device 
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](snr_batch).unsqueeze(1).unsqueeze(2).expand(-1, H,W, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        x = x.permute(0,3,1,2)
        return x

class NTC_Hyperprior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ga = ga(**config.ga_kwargs)
        self.gs = gs(**config.gs_kwargs)
        self.ha = ha()
        self.hs = hs()
        #-----------------------------------------------------------------
        self.num_slices = 8
        self.max_support_slices=8
        M=256
        self.window_size=8

        self.mamba_mean = nn.ModuleList(
            nn.Sequential(
                MambaAtten((M + (M//self.num_slices)*min(i, 8)), (M + (M//self.num_slices)*min(i, 8)), inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.mamba_scale = nn.ModuleList(
            nn.Sequential(
                MambaAtten((M + (M//self.num_slices)*min(i, 8)), (M + (M//self.num_slices)*min(i, 8)), inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M//self.num_slices)*min(i, 8), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M//self.num_slices)*min(i, 8), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M//self.num_slices)*min(i+1, 9), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        #-----------------------------------------------------------------
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.distortion = Distortion(config)
        self.H = self.W = 0
        # self.num_layer = len(config.gs_kwargs['embed_dims'])

        

    # def update_resolution(self, H, W):
    #     if H != self.H or W != self.W:
    #         self.ga.update_resolution(H, W)
    #         self.gs.update_resolution(H // 16, W // 16)
    #         self.H = H
    #         self.W = W

    def forward(self, input_image, require_probs=False):
        B, C, H, W = input_image.shape
        # self.update_resolution(H, W)
        # print("x.shape=",input_image.shape)
        y = self.ga(input_image)
        # print("y.shape=",y.shape)
        y_shape = y.shape[2:]
        z = self.ha(y)
        # print("z.shape=",z.shape)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        #获取高斯分布参数
        # scales_hat, means_hat = self.hs(z_hat)
        gaussian_params = self.hs(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        # print("z_hat.shape=",scales_hat.shape)
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([means_hat] + support_slices, dim=1)
            #--------------------------------------------------------------
            mean_support = self.mamba_mean[slice_index](mean_support)
            #--------------------------------------------------------------
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([scales_hat] + support_slices, dim=1)
            #--------------------------------------------------------------
            scale_support = self.mamba_scale[slice_index](scale_support)
            #--------------------------------------------------------------
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        # print("y_hat.shape=",y_hat.shape)
        x_hat = self.gs(y_hat)
        
        mse_loss = self.distortion(input_image, x_hat)
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        if require_probs:
            return mse_loss, bpp_y, bpp_z, x_hat, y, y_likelihoods, scales, means
        else:
            return mse_loss, bpp_y, bpp_z, x_hat

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss
    
class NTSCC_Hyperprior(NTC_Hyperprior):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.channel = Channel(config)
        self.fe = JSCCEncoder(**self.config.fe_kwargs)
        self.fd = JSCCDecoder(**self.config.fd_kwargs)
        if config.use_side_info:
            embed_dim = 256
            self.hyprior_refinement = Mlp(embed_dim * 3, embed_dim * 6, embed_dim)
        self.eta = config.eta
        self.snr_enc=snr_net(256)
        self.snr_dec=snr_net(256)
    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # B C H W
        return likelihoods


    def forward(self, input_image,snr, **kwargs):
        B, C, H, W = input_image.shape
        num_pixels = H * W * 3 * B
        # self.update_resolution(H, W)

        # time=Timer()
        # time.start()
        mse_loss_ntc, bpp_y, bpp_z, x_hat_ntc, y, y_likelihoods, scales_hat, means_hat = \
            self.forward_NTC(input_image, require_probs=True)
        # print("Ntc_time",time.end())
        y_likelihoods = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)

        # DJSCC forward
        # time.start()

        s_masked, mask_BCHW, indexes = self.fe(y, y_likelihoods.detach(), eta=self.eta)
        # print("s.shape=",s_masked.shape)
        #通过snr_net
        y_snr=self.snr_enc(s_masked,snr)

        # Pass through the channel.
        mask_BCHW = mask_BCHW.bool()
        channel_input = torch.masked_select(y_snr, mask_BCHW)
        # print("Encode_time",time.end())
        channel_output, channel_usage = self.channel.forward(channel_input,snr)
        s_hat = torch.zeros_like(y_snr)
        s_hat[mask_BCHW] = channel_output
        cbr_y = channel_usage / num_pixels


        y_hat_snr=self.snr_dec(s_hat,snr)

        # time.start()
        y_hat = self.fd(y_hat_snr, indexes)
        # print("y_hat.shape=",y_hat.shape)
        # hyperprior-aided decoder refinement (optional)
        if self.config.use_side_info:
            y_combine = torch.cat([BCHW2BLN(y_hat), BCHW2BLN(means_hat), BCHW2BLN(scales_hat)], dim=-1)
            y_hat = BLN2BCHW(BCHW2BLN(y_hat) + self.hyprior_refinement(y_combine), H // 4, W // 4)
        x_hat_ntscc = self.gs(y_hat).clip(0, 1)
        # print("x_hat.shape=",x_hat_ntscc.shape)
        # print("Decode_time",time.end())

        mse_loss_ntscc = self.distortion(input_image, x_hat_ntscc)
        return mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc

    def forward_NTC(self, input_image, **kwargs):
        return super(NTSCC_Hyperprior, self).forward(input_image, **kwargs)
