import torch
from torch import nn, einsum
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange

def exists(val):
    return val is not None

class SinusoidalEmbeddings(Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k

# ----------------------------------------------------------------

import math

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

# from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1,1)
    return tensors.movedim(-1,dim).flatten(dim, dim + 1)

# main class

class LocalAttention(Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False,
        scale = None,
        use_rotary_pos_emb = True,
        use_xpos = False,
        xpos_scale_base = None
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos = use_xpos,
                scale_base = default(xpos_scale_base, window_size // 2)
            )

    def forward(
        self,
        q, k, v,
        mask = None,
        input_mask = None,
        attn_bias = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out
    
# ----------------------------------------------------------------

import torch
from torch import nn, einsum
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange

def exists(val):
    return val is not None

class SinusoidalEmbeddings(Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k

# ------------------------------------------------------------------


import math

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

# from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1,1)
    return tensors.movedim(-1,dim).flatten(dim, dim + 1)

# main class

class LocalAttention(Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False,
        scale = None,
        use_rotary_pos_emb = True,
        use_xpos = False,
        xpos_scale_base = None
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos = use_xpos,
                scale_base = default(xpos_scale_base, window_size // 2)
            )

    def forward(
        self,
        q, k, v,
        mask = None,
        input_mask = None,
        attn_bias = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out

# -------------------------------------------------------------------------

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(Module):
    def __init__(
        self,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out

# ----------------


class LocalMHA(Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.causal = causal
        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = self.exact_windowsize,
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        attn_bias = None,
        cache = None,
        return_cache = False
    ):
        seq_len = x.shape[-2]

        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        if exists(cache):
            assert seq_len == 1

            assert self.causal and not exists(mask), 'only allow caching for specific configuration'

            ck, cv = cache

            q = q * (q.shape[-1] ** -0.5)

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            effective_window_size = self.attn_fn.look_backward * self.window_size

            if self.exact_windowsize:
                kv_start_index = -(effective_window_size + 1)
            else:
                seq_len = k.shape[-2]
                kv_start_index = -(effective_window_size + (seq_len % self.window_size))

            k, v = tuple(t[..., kv_start_index:, :] for t in (k, v))

            if exists(self.attn_fn.rel_pos):
                rel_pos = self.attn_fn.rel_pos
                pos_emb, xpos_scale = rel_pos(k)
                q, k = apply_rotary_pos_emb(q, k, pos_emb, scale = xpos_scale)

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(attn_bias):
                k_len = k.shape[-2]
                attn_bias = attn_bias[..., -1:, -k_len:]
                assert attn_bias.shape[-1] == sim.shape[-1]
                sim = sim + attn_bias

            attn = sim.softmax(dim = -1)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        else:
            out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if return_cache:
            kv = torch.stack((k, v))

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, kv

# -------------------------------------------------------------------------

from math import log2, ceil
from functools import partial
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from simple_hierarchical_transformer.attention import Attend

from typing import Tuple
# from local_attention import LocalMHA

# constants

Linear = partial(nn.Linear, bias = False)

LocalMHA = partial(LocalMHA, causal = True, prenorm = True)

# helper functions

def exists(val):
    return val is not None

def is_power_of_two(n):
    return log2(n).is_integer()

def all_unique(arr):
    return len(set(arr)) == len(arr)

def apply_fns(fns, tensors):
    return [fn(tensor) for fn, tensor in zip(fns, tensors)]

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def cosine_sim_loss(x, y):
    x, y = map(l2norm, (x, y))
    return 1. - einsum('b n d, b n d -> b n', x, y).mean()

# sampling helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        use_xpos = True
    ):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    @property
    def device(self):
        return next(self.buffers()).device

    @autocast(enabled = False)
    def forward(self, seq_len):
        device = self.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_2(pos, t, scale = 1.):
    seq_len = t.shape[-2]

    pos = pos[..., -seq_len:, :]
    if not isinstance(scale, (int, float)):
        scale = scale[..., -seq_len:, :]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


@autocast(enabled = False)
def apply_rotary_pos_emb_qk(rotary_emb, q, k):
    freqs, scale = rotary_emb
    q = apply_rotary_pos_emb_2(freqs, q, scale)
    k = apply_rotary_pos_emb_2(freqs, k, scale ** -1)
    return q, k

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# hierarchy related classes

def pad_seq_to_multiple(t, mult):
    seq_len = t.shape[-2]
    next_seq_len_mult = ceil(seq_len / mult) * mult
    remainder = next_seq_len_mult - seq_len

    if remainder == 0:
        return t, seq_len

    t = F.pad(t, (0, 0, 0, remainder), value = 0.)
    return t, seq_len

def curtail_seq_to_multiple(t, mult):
    seq_len = t.shape[-2]
    prev_seq_len_mult = (seq_len // mult) * mult
    remainder = seq_len - prev_seq_len_mult

    if remainder == 0:
        return t

    t = t[..., :prev_seq_len_mult, :]
    return t

def hierarchical_cat(tokens, strides: Tuple[int, ...]):
    assert len(tokens) == len(strides)

    if all([s == 1 for s in strides]):
        return torch.cat(tokens, dim = -1)

    tokens = [repeat(t, 'b n d -> b (n s) d', s = s) for t, s in zip(tokens, strides)]
    min_seq_len = min([t.shape[-2] for t in tokens])
    tokens = [t[..., :min_seq_len, :] for t in tokens]
    return torch.cat(tokens, dim = -1)

class CausalConv(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride = 1
    ):
        super().__init__()
        self.causal_padding = kernel_size - 1
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, stride = stride)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)

class Compress(Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        num_tokens = None,
        stride = 1,
        compress_factor = 1,
        expansion_factor = 4,
        dim_head = 64,
        heads = 8,
        ignore_index = 0,
        should_recon = False
    ):
        super().__init__()
        assert compress_factor > 0 and is_power_of_two(compress_factor)

        self.stride = stride
        self.no_compress = compress_factor == 1
        self.compress_factor = compress_factor

        self.should_recon = should_recon

        if self.no_compress:
            self.compress_fn = Linear(dim, dim_out) if dim != dim_out else nn.Identity()
            return

        dim_inner = int(dim * expansion_factor)

        self.compress_fn = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv(dim, dim_inner, compress_factor, stride = stride),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim_out, 1),
            Rearrange('b d n -> b n d')
        )

        if should_recon:
            assert exists(num_tokens)
            self.to_recon = Linear(dim_out, compress_factor * num_tokens)

        self.ignore_index = ignore_index

    def recon(self, h, ids):
        assert self.should_recon

        if self.no_compress:
            return torch.zeros((), device = h.device).requires_grad_()

        c = self.compress_factor
        seq_len = ids.shape[-1]

        recon_logits = self.to_recon(h)
        recon_logits = rearrange(recon_logits, 'b n (c d) -> (b c) d n', c = c)

        recon_ids = F.pad(ids, (c - 1, 0), value = self.ignore_index)
        recon_ids = tuple(recon_ids[:, i:(seq_len + i)] for i in range(c))
        recon_ids = torch.stack(recon_ids, dim = 1)
        recon_ids = rearrange(recon_ids, 'b c n -> (b c) n')

        if self.stride > 1:
            recon_ids = recon_ids[..., ::self.stride]

        recon_loss = F.cross_entropy(recon_logits, recon_ids, ignore_index = self.ignore_index)
        return recon_loss

    def forward(self, x):
        return self.compress_fn(x)

class HierarchicalMerge(Module):
    def __init__(
        self,
        dims: Tuple[int, ...],
        dim_out,
        h_strides = 1
    ):
        super().__init__()
        dim = sum(dims)

        strides = cast_tuple(h_strides, len(dims))
        assert len(strides) == len(dims)

        self.strides = strides

        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_out * 2),
            nn.SiLU(),
            nn.Linear(dim_out * 2, dim_out)
        )

    def forward(self, tokens):
        x = hierarchical_cat(tokens, self.strides)
        return self.net(x)

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        dim_inner = int(dim * mult)

        self.net = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_inner),
            nn.GELU(),
            Linear(dim_inner, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        use_flash_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = True, use_flash_attn = use_flash_attn)

        self.to_qkv = Linear(dim, dim_inner * 3)
        self.to_out = Linear(dim_inner, dim)

    def forward(self, x):
        n = x.shape[-2]
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        rotary_emb = self.rotary_emb(n)
        q, k = apply_rotary_pos_emb_qk(rotary_emb, q, k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class HierarchicalBlock(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        window_size = None,
        compress_factor = 1,
        stride = 1,
        ff_mult = 4
    ):
        super().__init__()
        self.stride = stride

        assert is_power_of_two(compress_factor)
        self.compress_factor = compress_factor
        self.no_compress = compress_factor == 1

        assert not exists(window_size) or window_size >= 0
        self.has_attn = window_size != 0

        self.attn = None

        if self.has_attn:
            attn_klass = Attention
            if exists(window_size):
                attn_klass = partial(LocalMHA, window_size = window_size)

            self.attn = attn_klass(dim = dim, dim_head = dim_head, heads = heads)

        self.ff = FeedForward(dim = dim, mult = ff_mult)

    def forward(self, x):
        c = self.compress_factor
        axial_dim = c // self.stride

        x, orig_seq_len = pad_seq_to_multiple(x, axial_dim)

        # hierarchical attention is performed with a simple axial attention

        # this, and using a convolution for compressing at the beginning
        # is one of the improvements on top of hourglass transformer
        # the downside is that the savings are only O(c) instead of O(c ** 2) as in hourglass transformer
        # you can get the O(c ** 2) saving by setting the hierarchical stride == c, but you'll see that performance is much worse, as some tokens will have a c - 1 token gap to the last hierarchical token

        if not self.no_compress:
            x = rearrange(x, 'b (n c) d -> (b c) n d', c = axial_dim)

        if exists(self.attn):
            x = self.attn(token_shift(x)) + x

        x = self.ff(token_shift(x)) + x

        if not self.no_compress:
            x = rearrange(x, '(b c) n d -> b (n c) d', c = axial_dim)

        return x[:, :orig_seq_len]

class HierarchicalTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        seq_len = 2048,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        hierarchies = 1,
        window_sizes = None,
        hierarchical_stride = 1,
        hierarchy_merge_all = False,  # whether to pass the pooled hierarchical information back to all hierarchies or just one doing the prediction
        predict_hierarchy = None,
        predict_use_all_hierarchy = False,
        recon_loss_weight = 0.1,
        hierarchical_ar_loss_weight = 0.25,
        ignore_index = 0,
        use_flash_attn = False,
    ):
        super().__init__()
        self.seq_len = seq_len

        hierarchies = cast_tuple(hierarchies)
        assert all_unique(hierarchies), 'hierarchies compression factors must be all unique integers'
        assert all([*map(is_power_of_two, hierarchies)]), 'only powers of two allowed for hierarchies'

        self.hierarchies = hierarchies

        # just use a simple tuple list per hyperparameter to customize each hierarchy

        num_hierarchies = len(hierarchies)

        dims = cast_tuple(dim, num_hierarchies)
        assert len(dims) == num_hierarchies

        window_sizes = cast_tuple(window_sizes, num_hierarchies)
        assert len(window_sizes) == num_hierarchies

        dim_head = cast_tuple(dim_head, num_hierarchies)
        assert len(dim_head) == num_hierarchies

        heads = cast_tuple(heads, num_hierarchies)
        assert len(heads) == num_hierarchies

        ff_mult = cast_tuple(ff_mult, num_hierarchies)
        assert len(ff_mult) == num_hierarchies

        hierarchical_stride = cast_tuple(hierarchical_stride, num_hierarchies)

        assert all([*map(is_power_of_two, hierarchical_stride)]), 'all hierarchical strides must be power of two'
        assert all([s <= h for s, h in zip(hierarchical_stride, hierarchies)]), 'all strides must be less than the compression factor of the hierarchy'

        self.h_strides = hierarchical_stride

        assert len(hierarchical_stride) == num_hierarchies

        # this determines to which hierarchy is everything pooled into for final prediction
        # however, final next token prediction can also use all hierarchies with `predict_use_all_hierarchy`

        predict_hierarchy = default(predict_hierarchy, min(hierarchies))
        self.predict_hierarchy_index = hierarchies.index(predict_hierarchy)
        hierarchy_predict_dim = dims[self.predict_hierarchy_index]

        self.hierarchy_merge_all = hierarchy_merge_all
        assert hierarchy_merge_all or self.h_strides[self.predict_hierarchy_index] == 1, 'the hierarchy level being used for final next token prediction must have compression stride of 1'

        # training related loss weights

        self.recon_loss_weight = recon_loss_weight

        should_recon = recon_loss_weight > 0

        self.should_recon = should_recon

        # token embedding

        dim_token_emb = max(dims)
        self.token_emb = nn.Embedding(num_tokens, dim_token_emb)

        # hierarchy ar loss - following the same scheme as done in mirasol paper - cosine sim of prediction to next embedding

        self.hierarchical_ar_loss_weight = hierarchical_ar_loss_weight
        self.has_hierarchical_ar_loss = hierarchical_ar_loss_weight > 0.

        self.to_hierarchical_preds = ModuleList([])

        for dim, hierarchy in zip(dims, hierarchies):
            linear_pred = nn.Linear(dim, dim) if hierarchy > 1 else None                

            self.to_hierarchical_preds.append(linear_pred)

        # hierarchy compressions - 1x just uses the base token_emb weights

        self.compressors = ModuleList([])

        for dim, hierarchy, stride in zip(dims, hierarchies, hierarchical_stride):
            self.compressors.append(Compress(
                dim = dim_token_emb,
                dim_out = dim,
                num_tokens = num_tokens,
                compress_factor = hierarchy,
                stride = stride,
                should_recon = should_recon
            ))

        # post token embedding norms

        self.post_token_emb_norms = ModuleList([nn.LayerNorm(dim) for dim in dims])

        # layers

        self.layers = ModuleList([])

        self.dims = dims

        self.hierarchical_merges = ModuleList([])
        self.need_hierarchical_merge = num_hierarchies > 1

        for _ in range(depth):
            hierarchical_layer = ModuleList([])

            # add a transformer block for each layer in the hierarchy

            for hierarchy, h_stride, h_dim, h_window_size, h_dim_head, h_heads, h_ff_mult in zip(hierarchies, hierarchical_stride, dims, window_sizes, dim_head, heads, ff_mult):

                # make sure the window size never exceeds the effective sequence length

                effective_seq_len = seq_len // hierarchy

                if exists(h_window_size) and h_window_size > effective_seq_len:
                    print(f'window size for hierarchy {hierarchy}x is greater than effective sequence length - setting window size to None (which would use normal full attention)')
                    h_window_size = None

                # add attention and feedforward

                hierarchical_layer.append(
                    HierarchicalBlock(
                        dim = h_dim,
                        dim_head = h_dim_head,
                        heads = h_heads,
                        window_size = h_window_size,
                        compress_factor = hierarchy,
                        stride = h_stride,
                        ff_mult = h_ff_mult
                    )
                )

            self.layers.append(hierarchical_layer)

            # for merging the information across hierarchies
            # for now, only one direction, from all hierarchies to the hierarchy that is being used to make predictions on, set by predict_hierarchy_index above

            if not self.need_hierarchical_merge:
                continue

            merge = HierarchicalMerge(
                dims = dims,
                dim_out = hierarchy_predict_dim if not self.hierarchy_merge_all else sum(dims),
                h_strides = hierarchical_stride
            )

            self.hierarchical_merges.append(merge)

        # final post-transformer norms, for all hierarchies

        self.norms = ModuleList([nn.LayerNorm(dim) for dim in dims])

        # to logit, for hierarchy set at predict_hierarchy_index, or all hierarchies

        self.predict_use_all_hierarchy = predict_use_all_hierarchy
        logit_dim_in = sum(dims) if predict_use_all_hierarchy else hierarchy_predict_dim

        self.to_logits = Linear(logit_dim_in, num_tokens)

        # training related loss parameters

        self.ignore_index = ignore_index

        self.register_buffer('zeros', torch.tensor(0.), persistent = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.0,
        filter_thres = 0.9,
        **kwargs
    ):
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.seq_len:], **kwargs)[:, -1]
            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature)
            sample = rearrange(sample, 'b -> b 1')
            out = torch.cat((out, sample), dim = -1)

        return out[:, t:]

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        ids,
        return_loss = False,
        return_hierarchical_token_embeds = False,
        return_hierarchical_embeds = False,
        ablate_hierarchical_merge = False
    ):
        """
        einops notation:

        b - batch
        n - sequence length
        c - compression factor
        d - dimension
        """

        # if training, predict next token in sequence

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # assert seq len

        assert ids.shape[-1] <= self.seq_len

        # get token embeddings, and pad to multiple of compression factor

        x = self.token_emb(ids)

        # for every hierarchy, compress token embeddings appropriately to the hierarchical embeddings

        tokens = []

        for compress in self.compressors:
            tokens.append(compress(x))

        # post embedding norms

        tokens = apply_fns(self.post_token_emb_norms, tokens)

        # if one wants all the compressed token embeds
        # just to investigate the space

        if return_hierarchical_token_embeds:
            return tokens

        # layers

        for layer, merge in zip_longest(self.layers, self.hierarchical_merges):

            tokens = apply_fns(layer, tokens)

            # pool the information all hierarchies
            # and then update the tokens that will be used to make the final autoregressive prediction

            if not self.need_hierarchical_merge or ablate_hierarchical_merge:
                continue

            pooled = merge(tokens)

            if self.hierarchy_merge_all:
                tokens = [(t + p[..., ::s, :]) for t, p, s in zip(tokens, pooled.split(self.dims, dim = -1), self.h_strides)]
            else:
                predict_tokens = tokens[self.predict_hierarchy_index]
                predict_tokens = predict_tokens + pooled
                tokens[self.predict_hierarchy_index] = predict_tokens

        # final normalized embeddings

        embeds = apply_fns(self.norms, tokens)

        # if one wants all the normalized hierarchical embeds

        if return_hierarchical_embeds:
            return embeds

        # select the hierarchical embeddings that will be doing the predicting

        if self.predict_use_all_hierarchy:
            predict_embed = hierarchical_cat(embeds, self.h_strides)
        else:
            predict_embed = embeds[self.predict_hierarchy_index]

        # logits for predicting next token

        logits = self.to_logits(predict_embed)

        if not return_loss:
            return logits

        # autoregressive loss (predictive coding)

        logits = rearrange(logits, 'b n c -> b c n')
        ce_loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        # reconstruction losses for hierarchy tokens

        recon_losses = self.zeros.requires_grad_()

        if self.should_recon:
            for compress, t in zip(self.compressors, embeds):
                recon_loss = compress.recon(t, ids)
                recon_losses = recon_losses + recon_loss

        # hierarchical ar loss

        hierarchical_ar_losses = self.zeros.requires_grad_()

        for h_embed, maybe_h_pred_linear in zip(embeds, self.to_hierarchical_preds):
            if not exists(maybe_h_pred_linear):
                continue

            h_pred = maybe_h_pred_linear(h_embed)
            h_ar_loss = cosine_sim_loss(h_pred[:, :-1], h_embed[:, 1:])

            hierarchical_ar_losses = hierarchical_ar_losses + h_ar_loss

        # total loss

        total_loss = ce_loss + \
                     recon_losses * self.recon_loss_weight + \
                     hierarchical_ar_losses * self.hierarchical_ar_loss_weight

        return total_loss, (ce_loss, recon_losses, hierarchical_ar_losses)

import torch

model = HierarchicalTransformer(
    num_tokens = 20000,                # number of tokens
    dim = 512,                         # model dimensions
    depth = 4,                         # depth
    dim_head = 64,                     # dimension per attention head
    heads = 8,                         # attention heads
    seq_len = 1024,                    # sequence lengths
    hierarchies = (1, 2, 8),           # hierarchies - here we have 1x (like in a regular transformer), then 2x and 8x compressed hierarchical tokens that undergo their own transformer blocks. information is pooled into one hierarchy at each layer
    window_sizes = (32, 64, None)      # local attention window sizes - the idea is that the higher hierarchies can pass distant information to the local one. None stands for full receptive field. Setting 0 would turn off attention at this hierarchy altogether (while token shift will still be in effect in each layer)
)

ids = torch.randint(0, 20000, (16, 1000))

loss, _ = model(ids, return_loss = True)
loss.backward()

# after much training

logits = model(ids)

print(logits.shape)


ids = torch.randint(0, 20000, (16, 887))

# after much training

logits = model(ids)

print(logits.shape)


