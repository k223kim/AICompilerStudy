import torch
import transformer_engine.pytorch as te


torch.manual_seed(0)
hidden_size = 128
batch_size = 2
head_num = 64
rotary_percent = 0.5
device = torch.device("cuda:0")
cu_seqlens = torch.tensor(
    [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048],
    dtype=torch.int32,
    device=device,
)
t = torch.rand(
    (cu_seqlens[-1], head_num, hidden_size),
    dtype=torch.float32,
    device = device
)
rotary_pos_emb = te.attention.RotaryPositionEmbedding(hidden_size, rotary_percent)
emb = rotary_pos_emb(cu_seqlens[-1])

torch_output = te.attention.apply_rotary_pos_emb(t, emb, tensor_format="thd", cu_seqlens=cu_seqlens)
print(torch_output)