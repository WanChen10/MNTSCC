import torch.nn as nn

class config:

    train_data_dir = '/home/dl/data_hard/datasets/DIV2K/train_f/'
    kodak_dir = '/home/dl/data_hard/datasets/Kodak24'
    clic_dir='/home/dl/data_hard/datasets/clic2022'
    cifar_dir='/home/dl/data_hard/datasets/cifar-10'
    batch_size = 2
    num_workers = 8

    epoch = 500
    
    print_step = 50
    logger = None
 
    # training details
    image_dims = None
    lr = 1e-4
    aux_lr = 1e-3
    distortion_metric ='MSE'#'MSE'

    use_side_info = True
    train_lambda = 20#训练偏置，增大此值会缩小压缩率
    eta = 0.2 #0.2-》0.02

    channel = {"type": 'awgn', 'chan_param': 10}
    multiple_rate = [16,32,48,64,80,96,102,118, 134, 160, 186, 192, 208, 224,240, 256]#16,
    resolution=None

    @classmethod
    def set_model_params(cls):
        if cls.resolution == 256:
            cls.ga_kwargs = dict(
                patch_size=2, in_chans=3,
                depths=[1, 1, 2, 4], embed_dim=[256, 256, 256, 256],
                d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
            )
            cls.gs_kwargs = dict(
                out_chans=3,
                depths=[4, 2, 1, 1], embed_dim=[256, 256, 256, 256],
                d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
            )
            cls.fe_kwargs = dict(
                embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=cls.multiple_rate, img_size=16,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
            )
            cls.fd_kwargs = dict(
                embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=cls.multiple_rate, 
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
            )
        elif cls.resolution == 32:
            
            cls.ga_kwargs = dict(
                patch_size=2, in_chans=3,
                depths=[2, 4], embed_dim=[128, 256],
                d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
            )
            cls.gs_kwargs = dict(
                out_chans=3,
                depths=[4, 2], embed_dim=[256, 128],
                d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
            )
            cls.fe_kwargs = dict(
                embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=cls.multiple_rate, img_size=8,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
            )
            cls.fd_kwargs = dict(
                embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=cls.multiple_rate, 
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
            )

    # if resolution =="256":
    #     ga_kwargs = dict(
    #         patch_size=2,in_chans=3,
    #         depths=[1, 1, 2, 4],embed_dim=[256, 256, 256, 256],
    #         d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #         norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False
    #     )
    #     gs_kwargs = dict(
    #         out_chans=3,
    #         depths=[4, 2, 1, 1], embed_dim=[256, 256, 256, 256],
    #         d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #         norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
    #     )
    #     fe_kwargs = dict(
    #         embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=multiple_rate,img_size=16,
    #         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
    #     )
    #     fd_kwargs = dict(
    #         embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=multiple_rate,img_size=16,
    #         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
    #     )
    # elif resolution =="32":
    #     ga_kwargs = dict(
    #         patch_size=2,in_chans=3,
    #         depths=[2, 4],embed_dim=[128, 256],
    #         d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #         norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False
    #     )
    #     gs_kwargs = dict(
    #         out_chans=3,
    #         depths=[4, 2], embed_dim=[256,128],
    #         d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #         norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False
    #     )
    #     fe_kwargs = dict(
    #         embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=multiple_rate,img_size=8,
    #         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
    #     )
    #     fd_kwargs = dict(
    #         embed_dim=256, depths=[1, 1, 1], norm_layer=nn.LayerNorm, rate_choice=multiple_rate,img_size=8,
    #         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, d_state=16,
    #     )