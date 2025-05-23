import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from einops import rearrange
from segm.model.decoder import DecoderLinear, DecoderLinear_one_patch

from torchvision.models import resnet50
from segm.model.distill import DistillableViT, DistillWrapper

from functools import partial

class Segmenter(nn.Module):
    def __init__(
        self,
        cnn_encoder,
        encoder,
        decoder,
        mid_decode,
        n_cls,
        cnn_distill
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn_distill
        if self.cnn:
            self.cnn_encoder = cnn_encoder
            self.cnn_decoder = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, n_cls, kernel_size=1, stride=1)
            )
        
        self.mid_decode = mid_decode
        if self.mid_decode:
            self.simple_decoder = DecoderLinear(n_cls, self.patch_size, 384)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        # print(im.shape)
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        
        x = self.encoder(im, return_features=True)
        cnn_patch_masks = None
        cnn_pix_masks = None
        if self.cnn:
            # print(im.shape)
            x_cnn = self.cnn_encoder(im)
            # print(x_cnn.shape)
            cnn_patch_masks = self.cnn_decoder(x_cnn)
            # print(cnn_patch_masks.shape)
            cnn_pix_masks = F.interpolate(cnn_patch_masks, size=(H, W), mode="bilinear")
            cnn_pix_masks = unpadding(cnn_pix_masks, (H_ori, W_ori))

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled_num
        if self.encoder.distilled:
            x_distilled = x[:, 1: num_extra_tokens] # 前面的 distill 的部分
            x_enc = x[:, num_extra_tokens:]
        else:
            x_enc = x[:, num_extra_tokens:]
        
        mid_patch_mask = None
        mid_pix_mask = None
        if self.mid_decode:
            mid_patch_mask, x_emb = self.simple_decoder(x_distilled if self.encoder.distilled else x_enc, (H, W))
            mid_pix_mask = F.interpolate(mid_patch_mask, size=(H, W), mode="bilinear")
            mid_pix_mask = unpadding(mid_pix_mask, (H_ori, W_ori))

        fin_patch_masks, cnn_dis_patch_masks, cls_seg_feat = self.decoder(x_enc, (H, W), distilled=False) #self.encoder.distilled
        # print(fin_patch_masks.shape)
        
        p_mid_patch_mask, p_x_emb = self.simple_decoder(x_enc, (H,W))

        fin_pix_masks = F.interpolate(fin_patch_masks, size=(H, W), mode="bilinear")
        # print(fin_pix_masks.shape)

        fin_pix_masks = unpadding(fin_pix_masks, (H_ori, W_ori))
        # print(fin_pix_masks.shape)
        # exit()
        cnn_dis_masks = None
        if cnn_dis_patch_masks is not None:
            cnn_dis_masks = F.interpolate(cnn_dis_patch_masks, size=(H, W), mode="bilinear")
            cnn_dis_masks = unpadding(cnn_dis_masks, (H_ori, W_ori))
        
        return {"fin_patch_masks": fin_patch_masks, 
                "fin_pix_masks": fin_pix_masks, 
                "cls_seg_feat": cls_seg_feat,
                "mid_patch_mask": mid_patch_mask, 
                "mid_pix_mask": mid_pix_mask, 
                "cnn_patch_masks": cnn_patch_masks,
                "cnn_pix_masks": cnn_pix_masks,
                "cnn_dis_patch_masks": cnn_dis_patch_masks, 
                "cnn_dis_masks": cnn_dis_masks,
                "p_mid_patch_mask": p_mid_patch_mask}

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)



class Segmenter_one_patch(nn.Module):
    def __init__(
        self,
        cnn_encoder,
        encoder,
        decoder,
        mid_decode,
        n_cls,
        cnn_distill
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn_distill
        if self.cnn:
            self.cnn_encoder = cnn_encoder
            self.cnn_decoder = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, n_cls, kernel_size=1, stride=1)
            )
        
        self.mid_decode = mid_decode
        if self.mid_decode:
            self.simple_decoder = DecoderLinear_one_patch(n_cls, self.patch_size, 384)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        
        x = self.encoder(im, return_features=True)
        cnn_patch_masks = None
        cnn_pix_masks = None
        if self.cnn:
            x_cnn = self.cnn_encoder(im)
            cnn_patch_masks = self.cnn_decoder(x_cnn)
            cnn_pix_masks = F.interpolate(cnn_patch_masks, size=(H, W), mode="bilinear")
            cnn_pix_masks = unpadding(cnn_pix_masks, (H_ori, W_ori))

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled_num
        if self.encoder.distilled:
            x_distilled = x[:, 1: num_extra_tokens] # 前面的 distill 的部分
            x_enc = x[:, num_extra_tokens:]
        else:
            x_enc = x[:, num_extra_tokens:]
        
        mid_patch_mask = None
        mid_pix_mask = None
        if self.mid_decode:
            x_emb = self.simple_decoder(x_distilled if self.encoder.distilled else x_enc, (H, W))
            # mid_pix_mask = F.interpolate(mid_patch_mask, size=(H, W), mode="bilinear")
            # mid_pix_mask = unpadding(mid_pix_mask, (H_ori, W_ori))

        fin_patch_masks, cnn_dis_patch_masks, cls_seg_feat = self.decoder(x_enc, (H, W), distilled=False) #self.encoder.distilled
        # print(fin_patch_masks.shape)
        

        fin_pix_masks = F.interpolate(fin_patch_masks, size=(H, W), mode="bilinear")
        # print(fin_pix_masks.shape)

        fin_pix_masks = unpadding(fin_pix_masks, (H_ori, W_ori))
        # print(fin_pix_masks.shape)
        # exit()
        cnn_dis_masks = None
        if cnn_dis_patch_masks is not None:
            cnn_dis_masks = F.interpolate(cnn_dis_patch_masks, size=(H, W), mode="bilinear")
            cnn_dis_masks = unpadding(cnn_dis_masks, (H_ori, W_ori))
        
        return {"fin_patch_masks": fin_patch_masks, 
                "fin_pix_masks": fin_pix_masks, 
                "cls_seg_feat": cls_seg_feat,
                "mid_patch_mask": mid_patch_mask, 
                "mid_pix_mask": mid_pix_mask, 
                "cnn_patch_masks": cnn_patch_masks,
                "cnn_pix_masks": cnn_pix_masks,
                "cnn_dis_patch_masks": cnn_dis_patch_masks, 
                "cnn_dis_masks": cnn_dis_masks,
                "one_cls_token": x_emb}

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
    
class Segmenter_one_patch_dconv(nn.Module):
    def __init__(
        self,
        cnn_encoder,
        encoder,
        decoder,
        mid_decode,
        n_cls,
        cnn_distill
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn_distill
        if self.cnn:
            self.cnn_encoder = cnn_encoder
            self.cnn_decoder = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, n_cls, kernel_size=1, stride=1)
            )
        
        self.dconv = nn.Sequential(
                nn.ConvTranspose2d(n_cls, n_cls, 4, stride=2, padding=1),
                nn.ConvTranspose2d(n_cls, n_cls, 4, stride=2, padding=1),
                nn.ConvTranspose2d(n_cls, n_cls, 4, stride=2, padding=1),
                nn.ConvTranspose2d(n_cls, n_cls, 4, stride=2, padding=1),
            )
        self.mid_decode = mid_decode
        if self.mid_decode:
            self.simple_decoder = DecoderLinear_one_patch(n_cls, self.patch_size, 384)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        
        x = self.encoder(im, return_features=True)
        cnn_patch_masks = None
        cnn_pix_masks = None
        if self.cnn:
            # print(im.shape)
            x_cnn = self.cnn_encoder(im)
            # print(x_cnn.shape)
            cnn_patch_masks = self.cnn_decoder(x_cnn)
            # print(cnn_patch_masks.shape)
            # print('-'*10)
            cnn_cls_token = torch.mean(rearrange(cnn_patch_masks, "b c h w -> b c (h w)"), 2)
            
            # print(cnn_patch_masks.shape)
            # exit()
            cnn_pix_masks = F.interpolate(cnn_patch_masks, size=(H, W), mode="bilinear")
            cnn_pix_masks = unpadding(cnn_pix_masks, (H_ori, W_ori))

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled_num
        if self.encoder.distilled:
            x_distilled = x[:, 1: num_extra_tokens] # 前面的 distill 的部分
            x_enc = x[:, num_extra_tokens:]
        else:
            x_enc = x[:, num_extra_tokens:]
        
        mid_patch_mask = None
        mid_pix_mask = None
        if self.mid_decode:
            # print("@@@", self.encoder.distilled)
            x_emb = self.simple_decoder(x_distilled if self.encoder.distilled else x_enc, (H, W))
            
            x_emb = x_emb[:,:,None,:]
            # print(x_emb.shape) # 16, 1, 1, 4
            x_emb = rearrange(x_emb, "b h w c -> b c h w")
            x_emb = self.dconv(x_emb)
            # print("x_emb.shape", x_emb.shape)
            mid_patch_mask = F.interpolate(x_emb, size=(14, 14), mode="bilinear")
            # print(mid_patch_mask.shape)
            # exit()
            mid_pix_mask = F.interpolate(mid_patch_mask, size=(H, W), mode="bilinear")
            mid_pix_mask = unpadding(mid_pix_mask, (H_ori, W_ori))



        fin_patch_masks, cnn_dis_patch_masks, cls_seg_feat = self.decoder(x_enc, (H, W), distilled=False) #self.encoder.distilled
        # print(fin_patch_masks.shape)
        

        fin_pix_masks = F.interpolate(fin_patch_masks, size=(H, W), mode="bilinear")
        # print(fin_pix_masks.shape)

        fin_pix_masks = unpadding(fin_pix_masks, (H_ori, W_ori))
        # print(fin_pix_masks.shape)
        # exit()
        cnn_dis_masks = None
        if cnn_dis_patch_masks is not None:
            cnn_dis_masks = F.interpolate(cnn_dis_patch_masks, size=(H, W), mode="bilinear")
            cnn_dis_masks = unpadding(cnn_dis_masks, (H_ori, W_ori))
        
        return {"fin_patch_masks": fin_patch_masks, 
                "fin_pix_masks": fin_pix_masks, 
                "cls_seg_feat": cls_seg_feat,
                "mid_patch_mask": mid_patch_mask, 
                "mid_pix_mask": mid_pix_mask, 
                "cnn_patch_masks": cnn_patch_masks,
                "cnn_pix_masks": cnn_pix_masks,
                "cnn_dis_patch_masks": cnn_dis_patch_masks, 
                "cnn_dis_masks": cnn_dis_masks,
                "one_cls_token": x_emb,
                "cnn_cls_token": cnn_cls_token,
                "x_enc": x_enc}

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)