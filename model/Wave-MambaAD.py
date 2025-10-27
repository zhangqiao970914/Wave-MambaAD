import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import numpy as np
from hilbert import decode, encode
from pyzorder import ZOrderIndexer
import numbers, math


def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

# ==========Decoder ==========
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
        return x

class SCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='sweep', ):
        super().__init__()
        size = int(size)  
        max_num = size ** dim 
        indexes = np.arange(max_num) 
        if 'sweep' == scan_type:  
            locs_flat = indexes 
        elif 'scan' == scan_type:
            indexes = indexes.reshape(size, size) 
            for i in np.arange(1, size, step=2):  
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1) 
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1)) 
            locs_flat = []
            for z in indexes:  
                r, c = zi.rc(int(z))  
                locs_flat.append(c * size + r)  
            locs_flat = np.array(locs_flat)      
        elif 'zigzag' == scan_type: 
            indexes = indexes.reshape(size, size) 
            locs_flat = []
            for i in range(2 * size - 1):  
                if i % 2 == 0:  
                    start_col = max(0, i - size + 1) 
                    end_col = min(i, size - 1) 
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])  
                else:  
                    start_row = max(0, i - size + 1)  
                    end_row = min(i, size - 1)  
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])  
            locs_flat = np.array(locs_flat)  

        elif 'diagonal' == scan_type:  
            locs_flat = []

            for i in range(size):
                locs_flat.append(i * size + i)

            for i in range(size):
                for j in range(i + 1, size):  # j > i
                    locs_flat.append(i * size + j)

            for i in range(size):
                for j in range(i):  # j < i
                    locs_flat.append(i * size + j)

            for i in range(size):
                for j in range(size):
                    if (i * size + j) not in locs_flat:
                        locs_flat.append(i * size + j)  

            locs_flat = np.array(locs_flat) 
        elif 'spiral' == scan_type: 
            locs_flat = self.generate_spiral_scan(size)

        elif 'hilbert' == scan_type:
            bit = int(math.log2(size)) 
            locs = decode(indexes, dim, bit)  
            locs_flat = self.flat_locs_hilbert(locs, dim, bit) 
        else:
            raise Exception('invalid encoder mode')  
        locs_flat_inv = np.argsort(locs_flat) 
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1) 
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1) 
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)  
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False) 
    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = [] 
        l = 2 ** num_bit 
        for i in range(len(locs)):  
            loc = locs[i]
            loc_flat = 0  
            for j in range(num_dim):  
                loc_flat += loc[j] * (l ** j)   
            ret.append(loc_flat)  
        return np.array(ret).astype(np.uint64)  
    def generate_spiral_scan(self, size):
        indexes = np.arange(size * size).reshape(size, size)
        locs_flat = []
        top, bottom, left, right = 0, size - 1, 0, size - 1

        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                locs_flat.append(indexes[top, i])
            top += 1
            for i in range(top, bottom + 1):
                locs_flat.append(indexes[i, right])
            right -= 1

            if top <= bottom:
                for i in range(right, left - 1, -1):
                    locs_flat.append(indexes[bottom, i])
                bottom -= 1
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    locs_flat.append(indexes[i, left])
                left += 1
        return np.array(locs_flat)

    def __call__(self, img):
        img_encode = self.encode(img)  
        return img_encode   
    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(img.shape), img) 
        return img_encode  
    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(img.shape), img) 
        return img_decode  


###### High-Frequency State Space ######
class D_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            size=4,
            #scan_type='sweep',
            num_direction=2,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))

        dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.num_direction)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.scans_h = SCANS(size=size, scan_type='sweep')
        self.scans_v = SCANS(size=size, scan_type='sweep')
        self.scans_d = SCANS(size=size, scan_type='diagonal')
        

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, scan_type: str):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.num_direction  # 获取方向数量 K
        xs = []

        if scan_type == 'horizontal':
            xs.append(self.scans_h.encode(x.view(B, -1, L)))
        elif scan_type == 'vertical':
            xs.append(self.scans_v.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)))
        elif scan_type == 'diagonal':
            xs.append(self.scans_d.encode(x.view(B, -1, L)))

        if K >= 2:
            xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)  # 重塑为 (B, K // 2, -1, L)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # 翻转并拼接

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

            xs = xs.float().view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
            Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
            Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
            Ds = self.Ds.float().view(-1)  # (k * d)
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
            assert out_y.dtype == torch.float

            inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
            ys = []
            if K >= 2:
                ys.append(self.scans_h.decode(out_y[:, 0])) 
                ys.append(self.scans_h.decode(inv_y[:, 0]))  
            y = sum(ys) 
            y = y.view(B, C, H, W)
            return y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)

        y_h, y_v, y_d = x.chunk(3, dim=0)
        y_h = self.forward_core(y_h, scan_type = 'horizontal')
        y_v = self.forward_core(y_v, scan_type = 'vertical')
        y_d = self.forward_core(y_d, scan_type = 'diagonal')

        y = torch.cat([y_h, y_v, y_d], dim=0)

        y = y.permute(0, 2, 3, 1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class ChannelMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dim=None,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_devide_out = if_devide_out
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.bimamba_type = bimamba_type
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(normalized_shape=dim)
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            bias=conv_bias,
            kernel_size=3,
            groups=dim,
            padding=1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

    def forward(self, u):
        """
        u: (B, H, W, C)
        Returns: same shape as hidden_states
        """
        b, d, h, w = u.shape #[b, h, 1, c]
        l = h * w
        u = rearrange(u, "b d h w-> b (h w) d").contiguous()
       
        conv_state, ssm_state = None, None
        
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(u, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=l,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)
        x = rearrange(self.conv2d(rearrange(x, "b l d -> b d 1 l")), "b d 1 l -> b l d")

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=l)
        B = rearrange(B, "(b l) d -> b d l", l=l).contiguous()
        C = rearrange(C, "(b l) d -> b d l", l=l).contiguous()

        x_dbl_b = self.x_proj_b(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = self.dt_proj_b.weight @ dt_b.t()
        dt_b = rearrange(dt_b, "d (b l) -> b d l", l=l)
        B_b = rearrange(B_b, "(b l) d -> b d l", l=l).contiguous()
        C_b = rearrange(C_b, "(b l) d -> b d l", l=l).contiguous()
        if self.bimamba_type == "v1":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A_b,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        elif self.bimamba_type == "v2":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out_b = selective_scan_fn(
                x.flip([-1]),
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            out = self.ln(out) * self.act(z)
            out_b = self.ln1(out_b) * self.act(z)
            if not self.if_devide_out:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w)
            else:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w) / 2

        return out

###### Low-Frequency State Space ######
class ChannelFreqAdapt(nn.Module):
    def __init__(self, dim, h, w, num_heads=4):
        super(ChannelFreqAdapt, self).__init__()
        self.num_heads = num_heads  

        self.CMamba_h = ChannelMamba(d_model=h, dim=dim // num_heads)
        self.CMamba_w = ChannelMamba(d_model=w, dim=dim // num_heads)

        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.conv1 = nn.Conv2d(dim // num_heads, dim // num_heads, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def channelmamba(self, x):
        x_h = self.pool_h(x).permute(0, 2, 3, 1)
        x_mamba_h = self.CMamba_h(x_h).permute(0, 3, 1, 2)
        x_mamba_h = x_mamba_h.expand_as(x) * x

        x_w = self.pool_w(x_mamba_h).permute(0, 3, 2, 1)
        x_mamba_w = self.CMamba_w(x_w).permute(0, 3, 2, 1)
        x_mamba_w = x_mamba_w.expand_as(x) * x
        out = self.conv1(x_mamba_w)
        return out

    def forward(self, x):
        res = x
        B, C, H, W = x.size()
        x = self.dwconv3(x)
        x_split = torch.chunk(x, self.num_heads, dim=1)
        out_list = []
        for i in range(self.num_heads):
            out_list.append(self.channelmamba(x_split[i]))

        y = self.project_out(torch.cat(out_list, dim=1))
        return y + res
    

########Dynamic Spatial Enhancement (DSE)##########
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x  
    def forward(self, x):
        x = self.project_in(x)
        x = self.channel_shuffle(x, 2)
        x_3 = self.dwconv3(x)
        x1, x2 = x_3.chunk(2, dim=1)
        x = F.gelu(x1) * x2 + F.gelu(x2) * x1
        x = self.project_out(x)
        return x

########Wave-Mamba Module##########
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        h: int = 0,
        w: int = 0,
        num_heads:int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        size: int = 4,
        num_direction=2,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.D_SSM = D_SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, size=size, num_direction=num_direction, **kwargs)
        self.CMamba = ChannelFreqAdapt(hidden_dim, h, w, num_heads)
        self.mlp = FeedForward(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.conv = nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim, kernel_size=1, stride=1)

        self.dwt = DWT()
        self.iwt = IWT()

    def forward(self, input: torch.Tensor): #[B, H, W, C]
        x = self.ln_1(input)

        y_wavelet = self.dwt(x.permute(0, 3, 1, 2).contiguous())  
        y_LL, y_HL, y_LH, y_HH = y_wavelet.chunk(4, dim=0)

        y_LL = self.CMamba(y_LL)
        High_f = torch.cat([y_HL, y_LH, y_HH], dim=0)
        High_f = self.D_SSM(High_f.permute(0, 2, 3, 1).contiguous())
        High_f = High_f.permute(0, 3, 1, 2).contiguous()
        
        y_HL, y_LH, y_HH = High_f.chunk(3, dim=0)

        enhanced_wavelet = torch.cat((y_LL, y_HL, y_LH, y_HH), dim=0)
        y = self.iwt(enhanced_wavelet)  
        y = y.permute(0, 2, 3, 1).contiguous()

        y1 = self.drop_path(y) + input
        y2 = self.ln_2(y1)

        y2 = self.mlp(y2.permute(0, 3, 1, 2).contiguous())
        y2 = y2.permute(0, 2, 3, 1).contiguous() + y1
        return y2


class ConvBNSSMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            h: int = 0,
            w: int = 0,
            num_heads:int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            depth: int = 2,
            size: int = 4,
            num_direction: int = 2,
            **kwargs,
    ):
        super().__init__()
        self.smm_blocks = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim, h=h, w=w, num_heads=num_heads, drop_path=drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, d_state=d_state, size=size, num_direction=num_direction,**kwargs)
            for i in range(depth)])

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        out_ssm = input  #[B, H, W, C]
        for blk in self.smm_blocks:
            out_ssm = blk(out_ssm)
        return out_ssm


class VSSLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            h,
            w,
            num_heads,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            size=4,
            num_direction=2,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint


        self.blocks = nn.ModuleList([
                ConvBNSSMBlock(
                    hidden_dim=dim,
                    h=h,
                    w=w,
                    num_heads=num_heads,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    depth=depth,
                    num_direction=num_direction,
                )
                for i in range(depth)])


        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

class MambaUPNet(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], h=[4, 8, 16, 32], w=[4, 8, 16, 32],  num_heads=[16, 8, 4, 2], depths_decoder=[2, 2, 2, 2],d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer = nn.LayerNorm, num_direction=2, ):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                h = h[i_layer],
                w = w[i_layer],
                num_heads = num_heads[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                size=4 * 2 ** (i_layer),
                num_direction=num_direction,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x): #[B,512,8,8]
        x = rearrange(x,'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x) #[B,8,8,512][B,16,16,256][B,32,32,128][B,64,64,64]
            if i != 0:
                out_features.insert(0, rearrange(x,'b h w c -> b c h w'))		
        return out_features #[B,64,64,64][B,128,32,32][B,256,16,16]

# ========== MFF & OCE ==========
class MFF_OCE(nn.Module):
    def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
        super(MFF_OCE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.SiLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):  # [B, 64, 64, 64] [B, 128, 32, 32] [16, 256,16,16 ]
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0 #[B,128,32,32]
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2]))) #[B,256,16,16]
        sv_features = self.relu(self.bnf(self.convf(sv_features))) #[B,256,16,16]
        sv_features = self.bn_layer(sv_features) #[B,512,8,8]
        return sv_features.contiguous()

class MAMBAAD(nn.Module):
    def __init__(self, model_t, model_s):
        super(MAMBAAD, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = MambaUPNet(depths_decoder=model_s['depths_decoder'])

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):  # [B,3,256,256]
        feats_t = self.net_t(imgs)  # [B,64,64,64]  [B,128,32,32]  [B,256,16,16]
        feats_t = [f.detach() for f in feats_t] # [B,64,64,64] [B,128,32,32] [B,256,16,16]
        feats_s = self.net_s(self.mff_oce(feats_t)) #[B,64,64,64] [B,128,32,32] [B,256,16,16]
        return feats_t, feats_s

@MODEL.register_module
def mambaad(pretrained=False, **kwargs):
    model = MAMBAAD(**kwargs)
    return model

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    vmunet = MambaUPNet([512, 256, 128, 64], [2, 2, 2, 2])
    bs = 1
    reso = 8
    x = torch.randn(bs, 512, reso, reso).cuda()
    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))