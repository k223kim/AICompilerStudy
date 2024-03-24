import torch
import transformer_engine.pytorch as te
import triton
import triton.language as tl
from pathlib import Path

MAX_FUSED_SIZE = 65536 # 2 ** 16; this is the maximum CUDA blocksize as per https://github.com/cuda-mode/lectures/blob/main/lecture3/pmpp.ipynb
GROUP_SIZE = 4

@triton.jit
def _rope_embedding_fwd(
    V,     V_stride,
    cos, cos_stride,
    sin, sin_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    head_start = group_head_position * GROUP_SIZE
    head_end = min((head_start + GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_position * V_stride + k * head_dim + col_offsets
        offs_q2 = row_position * V_stride + k * head_dim + col_offsets + half_head_dim

        V1 = tl.load(V + offs_q1, mask = mask, other = 0).to(tl.float32)
        V2 = tl.load(V + offs_q2, mask = mask, other = 0).to(tl.float32)

        tl.store(V + offs_q1, V1*cos1 - V2*sin1, mask = mask)
        tl.store(V + offs_q2, V2*cos1 + V1*sin1, mask = mask)

@triton.jit
def _rope_embedding_bwd(
    V,     V_stride,
    cos, cos_stride,
    sin, sin_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # the sign of sin has to be flipped!
    sin1 = -(tl.load(sin + (row_position % seqlen)*sin_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0))
    cos1 = tl.load(cos + (row_position % seqlen)*cos_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    head_start = group_head_position * GROUP_SIZE
    head_end = min((head_start + GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_position * V_stride + k * head_dim + col_offsets
        offs_q2 = row_position * V_stride + k * head_dim + col_offsets + half_head_dim

        V1 = tl.load(V + offs_q1, mask = mask, other = 0).to(tl.float32)
        V2 = tl.load(V + offs_q2, mask = mask, other = 0).to(tl.float32)

        tl.store(V + offs_q1, V1*cos1 - V2*sin1, mask = mask)
        tl.store(V + offs_q2, V2*cos1 + V1*sin1, mask = mask)

class RopeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, emb):
        cos, sin = emb.cos(), emb.sin()
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, hidden_dim = V.shape
        V = V.view(batch*seq_len, n_heads*hidden_dim)
        assert(seq_len <= cos.shape[0])

        half_hidden_dim = hidden_dim//2
        # below code is mostly from https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
        BLOCK_SIZE = triton.next_power_of_2(half_hidden_dim)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(f"{half_hidden_dim} exceeds the the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
        num_warps = 4
        if   BLOCK_SIZE >= 32768: num_warps = 32
        if BLOCK_SIZE >=  8192: num_warps = 16
        if BLOCK_SIZE >=  2048: num_warps = 8
        
        div, mod = divmod(n_heads, GROUP_SIZE)
        n_groups = div + (mod != 0)

        grid = (V.shape[0], n_groups, )
        _rope_embedding_fwd[grid](
              V,   V.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            hidden_dim, n_heads,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return V.view(batch, seq_len, n_heads, hidden_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, hidden_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*hidden_dim)

        cos = ctx.cos
        sin = ctx.sin

        grid = (dY.shape[0],ctx.n_groups, )
        _rope_embedding_bwd[(grid)](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, hidden_dim, n_heads,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, hidden_dim)
        return dY, None, None,

def get_embed(seq_length, hidden_size, rotary_percent):
    rotary_pos_emb = te.attention.RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)    
    return emb

def get_torch_output(input, emb, fused=True):
    torch_output = te.attention.apply_rotary_pos_emb(input, emb, tensor_format="bshd", fused=fused)
    return torch_output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size', 'seq_length', 'head_num', 'hidden_size'],  # Argument names to use as an x-axis for the plot
        x_vals=[i*2 for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['torch', 'torch-fused', 'triton'],
        # Label name for the lines
        line_names=["torch", "torch-fused", "triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="rope_embeddings",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))

def benchmark(batch_size, seq_length, head_num, hidden_size, provider):
    a = torch.randn((batch_size, seq_length, head_num, hidden_size), device='cuda', dtype=torch.float32)
    emb = torch.randn((seq_length, 1, 1, hidden_size), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: get_torch_output(a, emb, fused=False), quantiles=quantiles)
    if provider == 'torch-fused':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: get_torch_output(a, emb, fused=True), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: RopeTriton.apply(a, emb), quantiles=quantiles)
    perf = lambda ms: 2 * batch_size * seq_length * head_num * hidden_size * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    seq_length = 16 
    hidden_size = 128 
    rotary_percent = 1.0
    batch_size = 1
    head_num = 32
    dtype = torch.float32
    device = torch.device("cuda:0")
    input = torch.rand(
        (batch_size, seq_length, head_num, hidden_size),
        dtype=dtype,
        device = device
    )
    emb = get_embed(seq_length, hidden_size, rotary_percent)
    torch_output_fused = get_torch_output(input, emb, fused=True)
    torch_output = get_torch_output(input, emb, fused=False)
    triton_output = RopeTriton.apply(input, emb)
    # Pytorch 함수와 Triton 함수의 forward, backward 결과 값의 같다 (torch.testing.assert_close)는 것을 보여주는 코드
    torch.testing.assert_close(torch_output, triton_output)
    torch.testing.assert_close(torch_output_fused, triton_output)
    # Profiling한 결과와 개선한 점에 대한 설명
    benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())

