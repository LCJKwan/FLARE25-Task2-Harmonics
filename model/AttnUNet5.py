'''
Large Annotation Model
'''

import torch
from torch import nn
from torchvision.ops import stochastic_depth

class LayerNormTranspose(nn.Module):
    def __init__(self, dim: int, features: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(features, elementwise_affine=False, bias=False)

    def forward(self, x):
        # (..., C, ...) -> (..., ..., C) -> norm -> restore
        x = x.transpose(self.dim, -1)
        x = self.norm(x)
        return x.transpose(self.dim, -1)
    

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, 
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_c, h_c, 3, 1, 1, bias=bias),
            nn.GroupNorm(h_c, h_c),
            nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(h_c, h_c//2, 3, 1, 1, bias=bias),
            nn.GroupNorm(h_c//2, h_c//2),
            nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv3d(h_c, h_c//2, 3, 1, 2, dilation=2, bias=bias),
            nn.GroupNorm(h_c//2, h_c//2),
            nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c*2, out_c, 1, 1, 0, bias=False))
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x, self.conv2(x), self.conv3(x)], dim=1)
        return self.out_conv(x)


class ConvLayer(nn.Module):
    def __init__(self, in_c: int, conv: int, repeats: int, bias: bool = True, 
                 dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        self.repeats = repeats
        self.sto_depth = sto_depth
        self.convs = nn.ModuleList([
            ConvBlock(in_c, conv, in_c, bias, dropout)
            for _ in range(repeats)])

    def forward(self, x):
        for conv in self.convs:
            x = x + stochastic_depth(conv(x), self.sto_depth, 'row', self.training)
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_c, h_c * 2, bias)
        self.act = nn.SiLU()
        self.linear2 = nn.Sequential(
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(h_c, out_c, bias=False))
        
    def forward(self, x):
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        x = self.linear2(self.act(x1) * x2)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, in_c: int, head_dim: int, repeats: int, bias: bool = True,
                 dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        assert in_c % head_dim == 0, "in_c must be divisible by head_dim"
        self.sto_depth = sto_depth
        self.repeats = repeats
        self.mha_norms = nn.ModuleList([
            nn.LayerNorm(in_c) for _ in range(repeats)])
        self.MHAs = nn.ModuleList([
            nn.MultiheadAttention(in_c, in_c//head_dim, dropout=dropout, 
                        batch_first=True, bias=bias)
            for _ in range(repeats)])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_c),
                SwiGLU(in_c, in_c*2, in_c, bias=bias, dropout=dropout))
            for _ in range(repeats)])

    def forward(self, x):
        B, C, S1, S2, S3 = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, S1*S2*S3, C)
        for norm, mha, mlp in zip(self.mha_norms, self.MHAs, self.mlps):
            norm_x = norm(x)
            x = x + stochastic_depth(mha(norm_x, norm_x, norm_x, need_weights=False)[0], 
                                     self.sto_depth, 'row', self.training)
            x = x + stochastic_depth(mlp(x), self.sto_depth, 'row', self.training)
        x = x.permute(0, 2, 1).reshape(B, C, S1, S2, S3)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: list, convs: list, layers: list, dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        self.stages = len(channels)
        self.encoder_convs = nn.ModuleList(
            [nn.Sequential(
                ConvLayer(channels[i], convs[i], layers[i], bias=False, 
                          dropout=dropout * (i+1) / self.stages,
                          sto_depth=sto_depth * (i+1) / self.stages),
                nn.GroupNorm(channels[i]//8, channels[i]))
             for i in range(self.stages - 1)])
        self.downs = nn.ModuleList([nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False)
             for i in range(self.stages - 1)])
        
    def forward(self, x):
        skips = []
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x)
            skips.append(x)
            x = self.downs[i](x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, channels: list, convs: list, layers: list, dropout: float = 0.0, 
                 sto_depth: float = 0.0):
        super().__init__()
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        self.stages = len(channels)
        self.decoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, 
                       dropout=dropout * (i+1) / self.stages,
                       sto_depth=sto_depth * (i+1) / self.stages)
             for i in reversed(range(self.stages - 1))])
        self.ups = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i+1]//8, channels[i+1]),
                nn.ConvTranspose3d(channels[i+1], channels[i], 2, 2, 0, bias=False))
             for i in reversed(range(self.stages - 1))])
        self.merges = nn.ModuleList([
             nn.Conv3d(channels[i] * 2, channels[i], 3, 1, 1, bias=False)
             for i in reversed(range(self.stages - 1))])

    def forward(self, x, skips):
        for i, conv in enumerate(self.decoder_convs):
            x = self.ups[i](x)
            x = self.merges[i](torch.cat([x, skips.pop()], dim=1))
            x = conv(x)
        return x


class AttnUNet5(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        channels = p["channels"]
        convs = p["convs"]
        layers = p["layers"]
        head_dim = p["head_dim"]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"

        self.in_conv = nn.Conv3d(1, channels[0], (2, 2, 1), (2, 2, 1), 0, bias=False)
        
        self.encoder = Encoder(channels, convs, layers, dropout, sto_depth)
        self.bottleneck = nn.Sequential(
            *[nn.Sequential(
                ConvLayer(channels[-1], convs[-1], 1, 
                      bias=False, dropout=dropout, sto_depth=sto_depth),
                TransformerLayer(channels[-1], head_dim, 1,
                        bias=False, dropout=dropout, sto_depth=sto_depth))
                for _ in range(layers[-1])])
        self.decoder = Decoder(channels, convs, layers, dropout, sto_depth)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(channels[0], 16, (2, 2, 1), (2, 2, 1), 0, bias=False),
            LayerNormTranspose(1, 16),
            nn.Conv3d(16, out_c, 3, 1, 1, bias=True))

        
    def forward(self, x):
        x = self.in_conv(x)

        # Encoder
        x, skips = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, skips)

        x = self.out_conv(x)
        return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 224, 224, 112
    params = {
        "out_channels": 14,
        "channels":     [48, 96, 192, 384],
        "convs":        [32, 64, 96, 128],
        "head_dim":     64,
        "layers":       [6, 6, 6, 6],
        "dropout":      0.0,
        "stochastic_depth": 0.1
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = AttnUNet5(params).to(device)

    # Profile the forward and backward pass
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        # with torch.inference_mode():
        #     model.eval()
        #     # with torch.autocast('cuda', torch.bfloat16):
        #     y = model(x)
        with torch.autocast('cuda', torch.bfloat16):
            y = model(x)
            loss = y.sum()
        loss.backward()

    assert y.shape == (B, params["out_channels"], S1, S2, S3), "Output shape mismatch"
        
    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    if device == torch.device("cuda"):
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", round(total_params / 1e6, 2), 'M')
    
    # Calculate I/O sizes for input and output
    input_size_mb = x.element_size() * x.nelement() / 1024 / 1024
    output_size_mb = y.element_size() * y.nelement() / 1024 / 1024
    print("Input is size:", input_size_mb, 'MB')
    print("Output is size:", output_size_mb, 'MB')