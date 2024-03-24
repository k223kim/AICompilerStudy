# AICompilerStudy

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.10-py3
pip install git+https://github.com/NVIDIA/TransformerEngine.git@b8eea8a
```

or 

```bash
docker build . -t rope_triton
docker run --gpus all -it --rm rope_triton
```

## RoPE Embeddings

#### Forumla
As per the [official document](https://arxiv.org/abs/2104.09864) in essense, RoPE embeddings (RoPE for short) can be calculated by the following formula. Considering the input vector V, RoPE(V) can be expressed as the following.
```bash
RoPE(V) = cos * V + sin * _rotate_half(V)
```
Specifically, `_rotate_half` means that the input vector `V` is splitted into half with respect to its "head dimension" and the upper half and the bottom half are swapped. 

#### Back propagation of RoPE
Notice that when taking the derivative of the `RoPE` formula, the only interesting part is the `sin` part as it's sign changes to `-sin`. 

#### Group of Heads
The idea of 'group of heads' allows us to launch multiple heads in a single block which avoids the hassle to re-load the calculated sin and cos multiple times.