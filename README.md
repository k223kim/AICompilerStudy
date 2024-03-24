# AICompilerStudy

## Option1

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.10-py3
pip install git+https://github.com/NVIDIA/TransformerEngine.git@b8eea8a
```

## Option2

```bash
docker build . -t rope_triton
docker run --gpus all -it --rm rope_triton
```
and 
```bash
git clone https://github.com/k223kim/AICompilerStudy.git
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

#### Output comparison
This code passes the following two tests:
```python
# see line # 194, 195 for details
torch.testing.assert_close(torch_output, triton_output)
torch.testing.assert_close(torch_output_fused, triton_output)
```

#### Performance comparison
```table
    batch_size     torch  torch-fused    triton
0          4.0  0.000025     0.000125  0.000056
1          6.0  0.000121     0.000506  0.000253
2          8.0  0.000381     0.001631  0.000727
3         10.0  0.000888     0.003906  0.001628
4         12.0  0.001841     0.008100  0.002700
5         14.0  0.003262     0.012571  0.004689
6         16.0  0.005565     0.025600  0.007111
7         18.0  0.008201     0.034172  0.009320
8         20.0  0.012019     0.044643  0.012019
9         22.0  0.015777     0.057191  0.015777
10        24.0  0.020369     0.072000  0.015805
11        26.0  0.022885     0.081255  0.022885
12        28.0  0.024500     0.100042  0.025543
13        30.0  0.023970     0.113002  0.031020
14        32.0  0.024381     0.119810  0.084781
15        34.0  0.022119     0.124287  0.037827
16        36.0  0.022941     0.131220  0.038594
17        38.0  0.022879     0.135751  0.046246
18        40.0  0.024038     0.138889  0.048544
19        42.0  0.023927     0.141338  0.051945
20        44.0  0.024661     0.140779  0.051192
21        46.0  0.024738     0.138810  0.062914
22        48.0  0.026050     0.150261  0.095119
23        50.0  0.025378     0.135634  0.071806
24        52.0  0.025777     0.137024  0.076366
25        54.0  0.025828     0.131806  0.083038
26        56.0  0.026349     0.139188  0.088926
27        58.0  0.026112     0.127760  0.093260
28        60.0  0.026312     0.129145  0.098110
29        62.0  0.008427     0.123333  0.103813
30        64.0  0.008853     0.114174  0.158300
```