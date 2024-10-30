import sys
from pprint import pformat
from typing import List

import encoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from monai.losses import LocalNormalizedCrossCorrelationLoss
from torch.cuda.amp import autocast


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, size=(96, 96, 96), mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
    

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


class MDM(nn.Module):
    def __init__(
            self, encoder, decoder,
            mask_ratio=0.6, densify_norm='identity', sbn=False, norm_pix_loss=True
    ):
        super().__init__()
        input_size, downsample_ratio = encoder.input_size, encoder.downsample_ratio
        self.downsample_ratio = downsample_ratio
        self.fmap_size = input_size // downsample_ratio
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_size * self.fmap_size * self.fmap_size * (1 - mask_ratio))
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.sbn = sbn
        self.norm_pix_loss = norm_pix_loss
        self.hierarchy = len(encoder.enc_feat_map_chs)

        self.spatial_trans = SpatialTransformer()
        self.ncc_loss = LocalNormalizedCrossCorrelationLoss(kernel_size=5)
        self.grad = Grad3d(penalty='l2')
        self.proj_intensity = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj_warp = nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
    
    def mask(self, B: int, device, generator=None):
        f: int = self.fmap_size
        idx = torch.rand(B, f * f * f, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, f * f * f, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, f, f, f)
    
    def forward(self, inp_bchwd: torch.Tensor, ref_bchwd: torch.Tensor, active_b1fff=None, amp=True):
        # step1. Mask
        if active_b1fff is None:     # rand mask
            active_b1fff: torch.BoolTensor = self.mask(inp_bchwd.shape[0], inp_bchwd.device)  # (B, 1, f, f, f)
        encoder._cur_active = active_b1fff    # (B, 1, f, f, f)
        active_b1hwd = active_b1fff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio, 3).repeat_interleave(self.downsample_ratio, 4)  # (B, 1, H, W, D)
        masked_bchwd = inp_bchwd * active_b1hwd
        
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        with autocast(enabled=amp, dtype=torch.bfloat16):
            fea_bcfffs: List[torch.Tensor] = self.encoder(masked_bchwd)
            fea_bcfffs.reverse()  # after reversion: from the smallest feature map to the largest
            
            # step3. Decode and reconstruct
            out_bchwd = self.decoder(fea_bcfffs)
            rec_bchwd = self.proj_intensity(out_bchwd)
            warp_bchwd = self.proj_warp(out_bchwd)

        inp, rec = self.patchify(inp_bchwd), self.patchify(rec_bchwd)   # inp and rec: (B, L = f*f*f, N = C*downsample_ratio**2)
        if self.norm_pix_loss:
            mean = inp.mean(dim=-1, keepdim=True)
            var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
            inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)
        
        non_active = active_b1fff.logical_not().int().view(active_b1fff.shape[0], -1)  # (B, 1, f, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

        out_warped_bchwd = self.spatial_trans(inp_bchwd, warp_bchwd)
        warp_loss = self.ncc_loss(out_warped_bchwd, ref_bchwd)
        smooth_loss = self.grad(warp_bchwd, inp_bchwd)
        semantic_loss = warp_loss + 0.5 * smooth_loss
        
        loss = 0.5 * recon_loss + 0.5 * semantic_loss
        return loss, recon_loss, smooth_loss, warp_loss
    
    def patchify(self, bchwd):
        p = self.downsample_ratio
        h = w = d = self.fmap_size
        B, C = bchwd.shape[:2]
        bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
        bchwd = torch.einsum('bchpwqdr->bhwdpqrc', bchwd)
        bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3))  # (B, f*f, 1*downsample_ratio**3)
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_ratio
        h = w = d = self.fmap_size
        B, C = bln.shape[0], bln.shape[-1] // p ** 3
        bln = bln.reshape(shape=(B, h, w, d, p, p, p, C))
        bln = torch.einsum('bhwdpqrc->bchpwqdr', bln)
        bchwd = bln.reshape(shape=(B, C, h * p, w * p, d * p))
        return bchwd

    def calculate_rec_loss(self, rec, target, mask): 
        target = target / target.norm(dim=1, keepdim=True)
        rec = rec / rec.norm(dim=1, keepdim=True)
        active = mask.int()
        rec_loss = (1 - (target * rec).sum(1)).unsqueeze(1)
        rec_loss = rec_loss.mul_(active).sum() / (active.sum() + 1e-8)

        return rec_loss
    
    def __repr__(self):
        return (
            f'\n'
            f'[MDM.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[MDM.structure]: {super(MDM, self).__repr__().replace(MDM.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'encoder.input_size': self.encoder.input_size,
            # dec
            'decoder.width': self.decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(MDM, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(MDM, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[MDM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
    
    def denorm_for_vis(self, normalized_im, mean, std):
        normalized_im = (normalized_im * std).add_(mean)
        return torch.clamp(normalized_im, 0, 1)
