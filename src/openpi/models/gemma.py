# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

这个文件是 openpi 里最核心的 Transformer 适配层之一。它不是单纯复刻一个普通 Gemma，
而是把 Gemma 改造成“多 expert 共享 attention”的形式，服务于 `src/openpi/models/pi0.py`。

在 pi0 / pi0.5 里，上游 `Pi0.__init__` 会这样构造它：

    _gemma.Module(configs=[paligemma_config, action_expert_config], ...)

也就是说，`configs[0]` 对应 PaliGemma 语言/视觉前缀 expert，
`configs[1]` 对应 action expert。`pi0.py` 会把图像和文本编码成 prefix tokens，
把 noisy action / state / timestep 编码成 suffix tokens，然后交给这个 Module 做统一 Transformer 前向。

读这个文件时要一直记住三个项目级上下文：

1. `embed_prefix()` 产生的是条件信息：图像 token + prompt token。
2. `embed_suffix()` 产生的是动作生成相关信息：pi0 里有 state token + action tokens；
   pi0.5 里通常只有 action tokens，timestep 通过 `adarms_cond` 注入。
3. `sample_actions()` 推理时会先把 prefix 跑一遍得到 KV cache，然后每个 flow step 只跑 suffix，
   所以本文件里的 `kv_cache` 不是抽象概念，而是 openpi 加速动作采样的关键。

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops  # einops: 主要负责张量维度重排，这里在 attention 头拆分/合并时会频繁用到。
import flax.linen as nn  # nn: 这里仍然使用的是 Flax Linen 风格模块，而不是 NNX。
import jax  # jax: 注意力、einsum、softmax、scan/remat 等都在这里实现。
import jax.numpy as jnp  # jnp: JAX 版本的 numpy，承担大多数张量运算。

import openpi.models.lora as lora  # lora: 这里的线性层/FFN/attention 投影都支持可选 LoRA 注入。
import openpi.shared.array_typing as at  # at: 项目自己的数组类型标注和运行时 typecheck 装饰器。
import openpi.training.sharding as sharding  # sharding: 在大模型训练时给激活加分片约束，方便多设备执行。

PALIGEMMA_VOCAB_SIZE = 257_152  # PaliGemma 词表大小；这里所有 expert 共用这套离散 token id 空间。


@dataclasses.dataclass
class Config:
    # Config 这个类是“某一路 Gemma expert 的结构说明书”。
    # 它本身不做计算，只保存 Transformer 的宽度、层数、attention 头数、FFN 宽度、LoRA 设置等超参数。
    #
    # 为什么需要它：
    # - `pi0.py::Pi0.__init__` 会先调用 `get_config()` 得到两份 Config。
    # - 第一份 Config 描述 PaliGemma expert，也就是图像/语言 prefix 那一路。
    # - 第二份 Config 描述 action expert，也就是 noisy action suffix 那一路。
    # - 后面的 `Module / Block / Attention` 都靠这些 Config 知道每一路 expert 该建多宽、多少层、多少头。
    #
    # 你可以把它理解成：模型搭积木之前，每种积木的尺寸表。
    width: int  # width: Transformer hidden size，也就是 token embedding / residual stream 的宽度。
    depth: int  # depth: 堆多少层 Transformer block。
    mlp_dim: int  # mlp_dim: FFN 中间层宽度。
    num_heads: int  # num_heads: query 头数。
    num_kv_heads: int  # num_kv_heads: key/value 头数；支持 GQA/MQA 结构。
    head_dim: int  # head_dim: 每个注意力头的通道维度。
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)  # lora_configs: 指定 attn / ffn 哪些子层启用 LoRA。


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    # get_config 的作用：
    # 根据一个字符串名字返回对应规模的 Gemma 配置。
    #
    # 它在 openpi 流程里的位置：
    # - `pi0_config.py` 里会指定 `paligemma_variant` 和 `action_expert_variant`。
    # - `pi0.py::Pi0.__init__` 读取这两个名字后调用本函数。
    # - 返回的 Config 会传给 `Module(configs=[paligemma_config, action_expert_config])`。
    #
    # 输入:
    # - variant: 模型规模名，例如 "gemma_2b" 或 "gemma_300m"。
    #
    # 输出:
    # - Config: 一个只描述结构、不包含真实权重的 dataclass。
    #
    # 为什么不是直接在 pi0.py 里写死：
    # - 这样可以把“模型结构选择”和“模型前向逻辑”分开。
    # - 也方便 pi0 / pi0.5 / LoRA 版本复用同一套 Transformer 实现。
    if variant == "dummy":
        return Config(  # dummy: 单元测试/调试用的小模型，结构最小，方便快速跑通流程。
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(  # gemma_300m: 这里作为 action expert 的常用规模。
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(  # gemma_2b: 这里通常作为 PaliGemma 文本主干的规模配置。
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(  # gemma_2b_lora: 主干保持 2B 结构，但允许在 attention/ffn 上加 LoRA 微调。
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(  # gemma_300m_lora: 动作 expert 的 LoRA 版配置。
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    # RMSNorm 这个类负责“每个 Transformer 子层之前的归一化”。
    #
    # 它在系统中的位置：
    # - `Block.__call__` 在 attention 前调用一次 RMSNorm。
    # - `Block.__call__` 在 FFN 前又调用一次 RMSNorm。
    # - `Module.__call__` 在所有 Transformer block 结束后，还会给每个 expert 做 final_norm。
    #
    # 它为什么重要：
    # - 普通 pi0 中，它就是标准 RMSNorm，用来稳定 Transformer hidden states 的尺度。
    # - pi0.5 中，它还承担“把 flow timestep 条件注入 action expert”的任务。
    # - 也就是说，pi0.5 的时间条件不是简单拼到 action token 后面，而是在这里调制每层网络。
    #
    # 读代码时的抓手：
    # - cond is None: 普通 RMSNorm。
    # - cond is not None: adaptive RMSNorm，额外产生 scale / shift / gate。
    @nn.compact
    def __call__(self, x, cond):
        # __call__ 是 RMSNorm 真正的前向计算函数。
        #
        # 它做的事情可以分三步看：
        # 1. 先对每个 token 的 hidden vector 做 RMS 归一化。
        # 2. 如果没有条件 cond，就只用可学习 scale 做普通缩放。
        # 3. 如果有条件 cond，就从 cond 生成 scale / shift / gate，实现 pi0.5 的条件调制。
        #
        # 谁调用它：
        # - `Block.__call__` 调它来规范化 attention / FFN 的输入。
        # - `Module.__call__` 调它做每个 expert 的最后归一化。
        #
        # 谁消费它的输出：
        # - normed_inputs 会继续送入 attention 或 FFN。
        # - gate 会被 `_gated_residual()` 使用，控制子层更新量写回 residual stream 的强度。
        # 输入:
        # - x: 当前 expert 的 hidden states，通常形状是 [batch, seq_len, width]。
        #      在 pi0.py 里，x 可能来自 prefix tokens，也可能来自 suffix/action tokens。
        # - cond: 条件向量，形状通常是 [batch, width] 或 None。
        #         pi0 中一般是 None；pi0.5 的 action expert 中是 timestep embedding。
        #
        # 输出:
        # - normed_inputs: 和 x 同形状的归一化/调制后 hidden states。
        # - gate: None 或形状 [batch, 1, width] 的门控向量，会在 `_gated_residual()` 里使用。
        # RMSNorm 在 openpi 里有两种工作状态：
        # 1. cond is None: 普通 RMSNorm，用于 pi0，或者 pi0.5 的 prefix/PaliGemma expert。
        # 2. cond is not None: adaptive RMSNorm，用于 pi0.5 的 action expert。
        #
        # 这里的 cond 来自 `pi0.py::embed_suffix()`：
        # timestep -> posemb_sincos -> time_mlp_in/out -> adarms_cond。
        # 所以 cond 表示“当前 flow matching 时间 t”，不是语言 prompt，也不是 robot state。
        dtype = x.dtype  # dtype: x 的原始精度，可能是 bfloat16；最后输出要转回这个 dtype 以匹配主干计算。
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # var: 每个 token 自己沿 hidden dim 求均方值，形状是 [batch, seq_len, 1]。
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # normed_inputs: RMS 归一化后的 x，形状不变。
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))  # scale: 学习到的逐通道缩放参数，初值是 0，所以实际缩放从 1+scale 开始。
            normed_inputs = normed_inputs * (
                1 + scale
            )  # 普通 RMSNorm 的输出只做逐通道 rescale，不引入 shift/gate。
            return normed_inputs.astype(dtype), None  # 返回 None，表示当前 block 不需要 gated residual。

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)  # modulation: 从条件向量一次性预测出 scale / shift / gate 三组调制量。
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)  # [:, None, :]: 在序列维插入长度 1，便于广播到 [b, t, d]。
        normed_inputs = normed_inputs * (1 + scale) + shift  # pi0.5 在这里通过 cond 对 norm 后特征做条件化调制。
        # 这里的 scale/shift 改变的是 action expert 对当前 denoising 时间的“理解方式”：
        # t 靠近 1 时输入更像噪声，t 靠近 0 时输入更像真实动作，action expert 需要不同的处理策略。
        # gate 则留到 residual 阶段，让模型控制当前 attention/FFN 更新量应该写入多少。
        return normed_inputs.astype(dtype), gate  # gate 会在残差连接阶段使用，不是这里立即乘上去。


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    # Embedder 这个类负责“把离散 token id 变成连续向量”。
    #
    # 它在 openpi 里主要服务第 0 个 expert，也就是 PaliGemma 语言/视觉前缀 expert。
    # 具体来说：
    # - 文本 prompt 先经过 tokenizer 变成整数 token ids。
    # - `pi0.py::embed_prefix()` 调用 `self.PaliGemma.llm(..., method="embed")`。
    # - `Module.embed()` 再调用这里的 `Embedder.encode()`。
    # - 最终得到可以和图像 tokens 拼接的 text embeddings。
    #
    # 它不负责 action：
    # - action 是连续控制量，不走文本词表。
    # - action 在 `pi0.py` 里通过 `action_in_proj` 投影成 action expert tokens。

    vocab_size: int  # vocab_size: 离散 token 词表大小。
    embed_dim: int  # embed_dim: 每个 token 被映射到的隐藏维度。

    def setup(self):
        # setup 的作用：
        # 创建 embedding table 参数，也就是一个形状为 [vocab_size, embed_dim] 的大矩阵。
        #
        # 为什么在 setup 里创建：
        # - Flax Linen 的惯例是把需要保存/训练的参数注册在 setup 或 compact call 里。
        # - 这样 checkpoint 才知道这个矩阵是模型参数的一部分。
        #
        # 后面谁用它：
        # - `encode()` 用 token id 当索引查这张表。
        # - `decode()` 用这张表的转置把 hidden states 投回 vocab logits。
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )  # input_embedding_table: 共享词嵌入矩阵；这里只给第一个 expert 建立嵌入表。

    def encode(self, x):
        # encode 的作用：
        # 把整数 token ids 查表成连续 embedding vectors。
        #
        # 在 openpi 里的使用场景：
        # - 只用于语言 prompt tokens。
        # - 输出会被 `pi0.py::embed_prefix()` 和图像 tokens 拼在一起，作为 prefix 条件。
        #
        # 输入是离散的 token id；输出是 Transformer 能处理的连续 hidden states。
        # 输入:
        # - x: token ids，形状通常是 [batch, token_len]。
        # 输出:
        # - token embeddings，形状 [batch, token_len, embed_dim]。
        # 在 openpi 中，`pi0.py::embed_prefix()` 会通过 `self.PaliGemma.llm(..., method="embed")`
        # 调到这里，把 prompt token ids 变成 prefix hidden states。
        x = self.input_embedding_table[(x,)]  # 根据 token id 查表，得到 [b, t, d] 嵌入。
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 按常见 Transformer 做法乘 sqrt(d_model)，保持尺度一致。
        return x  # 返回离散 token 的连续表示。

    def decode(self, x):
        # decode 的作用：
        # 把 hidden states 投回词表 logits，理论上用于文本生成。
        #
        # 在 openpi 当前 pi0/pi0.5 主流程里：
        # - 动作不是从 vocab logits 采样出来的。
        # - 动作是 `pi0.py::action_out_proj` 从 action expert hidden states 线性投影出来的。
        # - 所以这个函数更像是保留 Gemma/PaliGemma 通用接口，而不是动作生成主路径。
        # 输入:
        # - x: hidden states，最后一维是 embed_dim。
        # 输出:
        # - logits，最后一维是 vocab_size。
        # pi0/flow matching 主路径不依赖 decode，因为动作是通过 action_out_proj 输出，不是解码文本。
        return jnp.dot(x, self.input_embedding_table.T)  # 用 embedding matrix 转置做 tied decoding。


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    # Attention 这个类负责“一层 Transformer block 里的多头 self-attention”。
    #
    # 它和普通 LLM attention 最大的不同：
    # - 普通 LLM 通常只有一条 token 流。
    # - openpi 这里有多条 expert token 流：第 0 路是 PaliGemma prefix，第 1 路是 action suffix。
    # - 本类会先让每个 expert 用自己的参数算 q/k/v，再把所有 expert 的 token 沿序列维拼起来统一 attention。
    #
    # 它为什么这么设计：
    # - prefix 和 suffix 需要互相通信，尤其 action tokens 必须能看见图像/语言条件。
    # - 但 PaliGemma expert 和 action expert 又希望有各自不同的参数。
    # - 所以这里采用“不同 expert 各自投影 q/k/v，共享同一张 attention 图”的折中结构。
    #
    # 它还负责 KV cache：
    # - 推理时 prefix 先单独跑一遍，得到 prefix 的 k/v。
    # - 每个 denoising step 只计算 suffix 的 q/k/v，再拼上缓存的 prefix k/v。
    # - 这就是 `sample_actions()` 能避免重复编码图像/文本的原因。

    configs: Sequence[Config]  # configs: 一个 expert 一个配置；这里要求所有 expert 的注意力头结构兼容。

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # __call__ 是 attention 的真正前向计算。
        #
        # 它完成的流程是：
        # 1. 对每个非 None 的 expert 输入，分别用该 expert 自己的权重投影 q/k/v。
        # 2. 把所有 expert 的 q/k/v 沿 token 序列维拼接。
        # 3. 给 q/k 应用 RoPE 位置编码。
        # 4. 如果有 kv_cache，就把历史 prefix k/v 拼到当前 k/v 前面。
        # 5. 根据 attn_mask 做 masked attention。
        # 6. 再把拼接后的输出切回每个 expert 自己的序列段。
        #
        # 谁调用它：
        # - `Block.__call__` 每一层都会调用一次。
        #
        # 谁消费它的输出：
        # - `Block.__call__` 会把 attention 输出通过 `_gated_residual()` 写回原来的 xs。
        # 输入:
        # - xs: list[hidden_states | None]。
        #       xs[i] 是第 i 个 expert 的 token 表示，形状通常是 [batch, seq_len_i, width_i]。
        #       在 openpi 当前主路径里：
        #       xs[0] = prefix/PaliGemma tokens，xs[1] = suffix/action expert tokens。
        # - positions: 所有本轮 query token 的 RoPE 位置，形状 [batch, total_query_len]。
        #              如果本轮只跑 suffix，它就是 suffix positions；如果训练全量前向，就是 prefix+suffix positions。
        # - attn_mask: attention 可见性 mask，形状 [batch, 1, query_len, key_len]。
        #              它决定 suffix 能看 prefix，而 prefix 不反向看 suffix。
        # - kv_cache: None 或之前 prefix 前向缓存下来的 (keys, values)。
        #
        # 输出:
        # - out: list[hidden_states | None]，结构和 xs 对齐，每个 expert 拿回自己的 attention 输出。
        # - (k, v): 更新后的 key/value，可作为下一步的 KV cache。
        # xs 是一个 list，每个元素对应一个 expert 的 token hidden states：
        # - xs[0]: PaliGemma expert 的 tokens，通常是图像 + prompt prefix。
        # - xs[1]: action expert 的 tokens，通常是 state/action suffix 或 action suffix。
        #
        # 在 `compute_loss()` 训练时，pi0.py 会传入 [prefix_tokens, suffix_tokens]，
        # 所以两个 expert 都参与前向。
        #
        # 在 `sample_actions()` 推理时有两种调用：
        # - 先传 [prefix_tokens, None] 填好 prefix KV cache。
        # - 后续每个 Euler step 传 [None, suffix_tokens]，并复用 prefix KV cache。
        #
        # 这就是为什么这里要支持 x is None，也为什么要返回新的 kv_cache。
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        # 这 3 个断言的意义很重要：
        # 因为后面要把多个 expert 的 q/k/v 在序列维直接拼起来统一做 self-attention，
        # 所以头数和 head_dim 必须一致，否则没法共享同一套注意力计算。

        dtype = next(x.dtype for x in xs if x is not None)  # dtype: 拿第一个真实输入 expert 的 dtype 作为当前注意力计算的目标精度。

        qkvs = []  # qkvs: list of tuples，每个元素是当前 expert 的 (q, k, v)。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue  # x is None: 说明这一路 expert 这次不参与前向，比如 prefix-only / suffix-only 模式。
            # 每个 expert 都有自己的 q/k/v 投影参数，参数名通过 `_name(..., i)` 区分。
            # 第 0 个 expert 叫 q_einsum / kv_einsum，第 1 个 expert 叫 q_einsum_1 / kv_einsum_1。
            # 这让 PaliGemma 权重能直接加载到第 0 个 expert，同时 action expert 拥有独立参数。
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )  # qkv_einsum: 这是“标准多头注意力里一次性线性映射出 q/k/v”的版本。
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))  # 输出形状里 B=batch, S=seq, K=heads, H=head_dim。
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )  # q_einsum: 当 num_kv_heads < num_heads 时，query 单独投影，支持 GQA/MQA。
                q = q_einsum("BTD,NDH->BTNH", x)  # q: 当前 expert 的 query，形状 [batch, token_len, num_heads, head_dim]。
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )  # kv_einsum: key/value 头数更少时，共享同一组 kv 头来服务多组 query 头。
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)  # k/v: 当前 expert 的 key/value，形状 [batch, token_len, num_kv_heads, head_dim]。
                qkvs.append((q, k, v))  # 最后无论哪条分支，都整理成 (q, k, v) 三元组。

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))  # q/k/v: 把所有参与本轮前向的 expert 在 token 维拼成一条总序列。
        # 拼接后的意义是：prefix tokens 和 suffix/action tokens 可以在同一个 attention 图里交互。
        # 具体谁能看谁，不由拼接本身决定，而是由 `attn_mask` 决定。
        # `pi0.py::make_attn_mask()` 会保证 prefix 不反向依赖 suffix，而 suffix 可以看 prefix。

        q = _apply_rope(q, positions=positions)  # q 使用 RoPE 注入位置信息。
        q *= self.configs[0].head_dim ** -0.5  # 缩放 query，避免点积随 head_dim 变大而数值过大。

        k = _apply_rope(k, positions=positions)  # k 也必须用同样的 RoPE，才能形成相对位置编码效果。

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache  # kv_cache: 推理时 prefix 先跑过一次后缓存下来的 key/value。
            k = jnp.concatenate([cache_k, k], axis=1)  # 把旧缓存和这一步新 suffix 的 k 拼起来，形成完整上下文。
            v = jnp.concatenate([cache_v, v], axis=1)  # v 同理。
            # 注意这里 cache 只缓存 K/V，不缓存 Q：
            # 因为当前 suffix token 仍然需要产生新的 query 去“查询”prefix 和 suffix 上下文。
            # prefix 的 K/V 固定不变，所以可以复用；suffix 的 K/V 每个 denoising step 都会随着 x_t 改变。

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)  # G: 表示“每个 kv 头服务多少个 query 头”。
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)  # logits: query 和 key 点积后的注意力分数，形状 [B, K, G, T, S]。

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # 用非常小的负数把不可见位置压到 softmax 后接近 0。
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)  # masked_logits: 把不可见 key 位置替换成极小值。

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)  # probs: 每个 query 对所有 key 的注意力概率分布。

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)  # encoded: 注意力聚合后的多头表示。
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")  # encoded: 头维合并后形状 [batch, query_len, num_heads, head_dim]。

        out = []  # out: 最后再把“合并后统一计算”的输出按 expert 序列长度切回去。
        start = 0  # start/end: 在拼接后的总序列里定位每个 expert 自己那一段。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )  # out_einsum: 标准 attention 输出投影，把多头结果映射回 residual 宽度。
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))  # 只取当前 expert 自己那段序列。
                start = end  # 下一个 expert 从后面继续切。
            else:
                out.append(None)  # 保持返回结构与输入 xs 对齐。
        # 返回 list 的形式非常关键：
        # `pi0.py` 依赖 `(prefix_out, suffix_out)` 这种解包方式，
        # 并且只会对 suffix_out 的最后 action_horizon 个 token 做 action_out_proj。

        return out, (k, v)  # 同时返回更新后的 kv，供推理缓存继续复用。


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    # FeedForward 这个类表示“一层 Transformer block 里的 MLP/FFN 子层”。
    #
    # 它的职责：
    # - attention 负责 token 与 token 之间的信息交互。
    # - FeedForward 负责对每个 token 自己的 hidden vector 做非线性变换。
    # - 换句话说，attention 主要混合序列信息，FFN 主要提升每个 token 内部特征表达能力。
    #
    # 注意当前 openpi 的 Block 实际调用的是 `lora.FeedForward`，不是直接调用这个类。
    # 这个类保留了 Gemma/big_vision 风格的 gated FFN 实现，读它可以理解 FFN 的数学结构；
    # 但如果你追踪当前 pi0/pi0.5 的主路径，应该重点看 `Block.__call__` 里那段 `lora.FeedForward(...)`。

    features: int  # features: 输入输出维度，也就是 residual stream 宽度。
    hidden_dim: int  # hidden_dim: FFN 中间层维度，通常比 features 大很多。

    @nn.compact
    def __call__(self, x):
        # __call__ 的作用：
        # 对输入 hidden states 做 gated feed-forward 变换，输出一个和输入同形状的更新量。
        #
        # 输入:
        # - x: 某个 expert 的 token hidden states。
        #
        # 输出:
        # - outputs: FFN 子层产生的增量，不包含 residual 相加。
        #
        # 为什么不在这里加 residual：
        # - residual 需要结合 pi0.5 的 gate。
        # - gate 来自 RMSNorm，而不是 FFN 自己。
        # - 所以 residual 统一放在 `Block.__call__` 的 `_gated_residual()` 里处理。
        # 输入:
        # - x: 某个 expert 的 hidden states，形状 [batch, seq_len, features]。
        # 输出:
        # - outputs: FFN 更新量，形状仍然是 [batch, seq_len, features]。
        # 注意：这里不负责残差相加，残差在 Block.__call__ 里统一处理。
        dtype = x.dtype  # dtype: 保持和外面 residual 一致，避免精度混乱。
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)  # w_gating[0] 和 w_gating[1] 对应 gated-GELU FFN 的两条分支。
        ff_gate = jnp.dot(x, w_gating[0])  # ff_gate: 门控分支的线性投影，形状 [batch, seq_len, hidden_dim]。
        gate_value = nn.gelu(ff_gate)  # gate_value: GELU 后的门控值。

        ff1 = jnp.dot(x, w_gating[1])  # ff1: 内容分支的线性投影，形状 [batch, seq_len, hidden_dim]。
        activations = gate_value * ff1  # activations: 门控后的 FFN 中间激活。

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)  # w_linear: 把 hidden_dim 再映射回 residual 宽度。
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs  # FFN 不做残差，残差在外层 Block 里处理。


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    # Block 这个类表示“一层完整的 Transformer”。
    #
    # 一层 Block 的结构可以记成：
    # 1. RMSNorm
    # 2. Attention
    # 3. residual / gated residual
    # 4. RMSNorm
    # 5. FFN
    # 6. residual / gated residual
    #
    # 它在 openpi 里的特殊点：
    # - 它不是只处理一条 hidden state，而是处理 xs 这个 expert list。
    # - xs[0] 通常是 PaliGemma prefix expert。
    # - xs[1] 通常是 action expert。
    # - 每层都让这两路在 attention 里交互，但又保留各自的 norm / FFN / 投影参数。
    #
    # pi0.5 的 adaRMSNorm 也主要在这里生效：
    # - Block 在 attention 前和 FFN 前调用 RMSNorm。
    # - 如果 action expert 收到 time_emb，它就会被注入每一层。

    configs: tuple[Config, ...]  # configs: 每个 expert 一份配置，但所有 expert 在同一 block 深度上同步推进。

    dropout: float = 0.0  # dropout: 训练时可选的随机失活概率。
    dropout_bdims: tuple[int, ...] = ()  # dropout_bdims: 指定 dropout 共享哪些 batch 维。

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        # __call__ 是一层 Transformer block 的前向执行。
        #
        # 它在整个模型流程里的位置：
        # - `Module.setup()` 用 `nn.scan` 把这个 Block 重复 depth 次，形成完整 Transformer。
        # - 每次执行一层，都会更新 xs，也会更新这一层对应的 kv_cache。
        #
        # 直观理解：
        # - 输入 xs 是“当前层开始时每个 expert 的 token 表示”。
        # - 输出 xs 是“经过 attention 和 FFN 后每个 expert 的 token 表示”。
        # - 如果是在推理，它还顺手维护本层的 KV cache。
        # 输入:
        # - xs: list[hidden_states | None]，一个 expert 一份 token hidden states。
        # - kv_cache: None 或历史 K/V 缓存，推理时用来让 suffix 读取 prefix。
        # - positions: 当前 query tokens 的 RoPE 位置。
        # - attn_mask: 当前 query tokens 对 key tokens 的可见性规则。
        # - adarms_cond: list[condition | None]，和 xs/configs 对齐。
        # - deterministic: 是否关闭 dropout；推理时通常为 True。
        #
        # 输出:
        # - xs: list[hidden_states | None]，每个 expert 经过一层 Transformer block 后的结果。
        # - kv_cache: 这一层更新后的 K/V 缓存。
        # 一个 Block 就是一层 Transformer。
        # 这里的特殊之处是：同一层同时处理多个 expert 的 token 流。
        #
        # 对 pi0 来说，adarms_cond 通常是 [None, None]：
        # - prefix expert 普通 RMSNorm
        # - action expert 普通 RMSNorm
        #
        # 对 pi0.5 来说，adarms_cond 通常是 [None, time_emb]：
        # - prefix expert 不受 flow timestep 调制
        # - action expert 通过 time_emb 做 adaptive RMSNorm
        xs = sharding.activation_sharding_constraint(xs)  # 给激活加 sharding 约束，帮助大模型并行训练。
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x  # 无 dropout 时退化成恒等函数。

        attn = Attention(configs=self.configs, name="attn")  # attn: 当前 block 内的 self-attention 子层。

        pre_attn = []  # pre_attn: list，存 attention 前归一化后的 hidden states。
        gates = []  # gates: list，存 attention 子层 residual 的门控值。
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901  # pi0.5 就是在这里把 timestep 条件打进 attention 前归一化。
            pre_attn.append(x)
            gates.append(gate if x is not None else None)
        # 这里的 gate 如果来自 pi0.5 action expert，就会调节 attention 输出写回 residual 的强度。
        # 直觉上，模型可以根据当前 flow 时间 t 决定“这一层 attention 更新要更激进还是更保守”。

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)  # post_attn: 每个 expert 的 attention 更新量；kv_cache: 新的 K/V。
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)  # 训练时可选 dropout。
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]  # attention 残差；若有 gate，则变成条件门控残差。
        xs = sharding.activation_sharding_constraint(xs)

        out = []  # out: list，存 FFN 子层输出。
        gates = []  # gates: list，存 FFN 子层 residual 的门控值。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901  # pi0.5 也会在 FFN 前归一化这里注入条件。
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)  # FFN 同样支持 LoRA 版实现。
            out.append(x)
            gates.append(gate if x is not None else None)
        # attention 和 FFN 前各做一次 RMSNorm，因此 pi0.5 的时间条件每层会注入两次：
        # 一次影响 attention 子层，一次影响 FFN 子层。

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]  # FFN 残差也可能带条件 gate。
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache  # block 输出新的 hidden states，以及当前层更新后的缓存。


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]  # KVCache: (all layer keys, all layer values) 的缓存结构。
# KVCache 的作用：
# 它保存所有 Transformer 层的 key/value，主要用于 `pi0.py::sample_actions()` 推理加速。
#
# 为什么只缓存 K/V，不缓存 Q：
# - Q 来自当前要生成/更新的 suffix tokens，每个 denoising step 都会变，所以要重新算。
# - prefix 的 K/V 来自图像和语言条件，在同一次动作采样中不变，所以可以缓存复用。
#
# 形状里的 l 表示层数；b 表示 batch；_t 表示缓存的 token 长度；_k/_v 表示 kv heads；_h 表示 head dim。


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    # Module 是本文件的顶层模型类，也是 `pi0.py` 里 `self.PaliGemma.llm` 的真实实现。
    #
    # 它负责把前面定义的零件组装成完整 Transformer：
    # - Embedder: 负责第 0 个 expert 的文本 token embedding。
    # - Block: 被重复 depth 次，构成 Transformer 主干。
    # - final_norms: 每个 expert 最后各自做一次 RMSNorm。
    #
    # 它的接口为什么用 list：
    # - `embedded[0]` 是 PaliGemma/prefix expert 的 tokens。
    # - `embedded[1]` 是 action expert/suffix 的 tokens。
    # - 某一路可以传 None，表示这一轮不跑它。
    #
    # 这种设计直接服务 openpi 的两种模式：
    # - 训练: `[prefix_tokens, suffix_tokens]` 一起跑，方便计算 loss。
    # - 推理: 先 `[prefix_tokens, None]` 填 KV cache，再多次 `[None, suffix_tokens]` 采样动作。

    configs: Sequence[Config]  # configs: 每个 expert 一份 Config。当前 openpi 实际上常用 2 个 expert：PaliGemma + action expert。
    embed_dtype: str  # embed_dtype: token embedding / hidden states 的主计算精度。

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False  # adarms: 这里只是配置位，真正是否传 cond 由上层调用时的 adarms_cond 决定。

    def setup(self):
        # setup 的作用：
        # 在 Flax Linen 里注册这个 Module 需要的所有子模块。
        #
        # 它不是一次真正的数据前向，而是“搭模型骨架”：
        # - 建立第 0 个 expert 的 Embedder。
        # - 用 `nn.scan` 创建 depth 层 Block。
        # - 给每个 expert 建 final RMSNorm。
        #
        # 这里最容易迷糊的是 `nn.scan`：
        # 它不是在 batch 维扫描，而是在 Transformer 层数维扫描。
        # 也就是说，它把同一个 Block 模板展开成第 0 层、第 1 层、……第 depth-1 层，
        # 每层都有自己的参数。
        # 输入:
        # - 无显式输入；setup 根据 self.configs / self.embed_dtype / dropout 配置创建参数化子模块。
        # 输出:
        # - 没有 return；副作用是注册 embedder、layers、final_norms 等子模块。
        #
        # 这个 setup 不是数据前向，而是 Flax Linen 的模块构造阶段。
        # Module 是 `pi0.py` 中 `self.PaliGemma.llm` 的实际主体。
        # 它要同时满足三种 openpi 调用方式：
        # 1. `method="embed"`: 只把 prompt token id 查表成 prefix embeddings。
        # 2. `[prefix_tokens, suffix_tokens]`: 训练时一次性跑完整上下文。
        # 3. `[None, suffix_tokens] + kv_cache`: 推理采样时只跑 suffix，并读取缓存的 prefix K/V。
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        # 多 expert 可以宽度不同，但 block 深度必须一样，
        # 因为这里是“第 0 层一起算、第 1 层一起算……”的同步扫描结构。

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )  # embedder 只给第一个 expert 用；后续 expert 一般直接接收外部已经构造好的连续 token。
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )  # remat: 对 block 做梯度检查点，节省训练显存。
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},  # params 这一维沿 layer 扫描展开：每层 block 有自己的一份参数。
            split_rngs={"params": True, "dropout": True},  # 每层初始化参数和 dropout 都拿独立 rng，避免所有层参数相同。
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )  # scan: 把同一个 Block 沿“层数”维重复展开，形成整栈 Transformer。
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]  # 每个 expert 末尾还有各自的 final norm。

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        # embed 是 Module 暴露给 `pi0.py::embed_prefix()` 的“文本 embedding 快捷入口”。
        #
        # 它只做一件事：
        # - 把 prompt token ids 交给 `Embedder.encode()`。
        #
        # 为什么要单独暴露这个 method：
        # - `pi0.py` 想要先把图像 tokens 和文本 tokens 拼成 prefix。
        # - 这一步只需要 embedding，不需要跑完整 Transformer。
        # - 因此调用 `self.PaliGemma.llm(obs.tokenized_prompt, method="embed")` 更高效也更清晰。
        # 输入:
        # - tokens: PaliGemma tokenizer 产生的 prompt token ids。
        # 输出:
        # - embeddings: 第一个 expert 的连续 token embeddings。
        # 在 openpi 中，这个函数只负责 prefix 里的语言 token，不负责 action token。
        return self.embedder.encode(tokens).astype(self.embed_dtype)  # 把离散 token id 转成第一个 expert 的连续前缀表示。

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        # __call__ 是完整 Gemma Transformer 的前向入口。
        #
        # 它做的事情：
        # 1. 接收每个 expert 已经准备好的 continuous token embeddings。
        # 2. 整理 attention mask 的维度。
        # 3. 把 tokens 送入 depth 层 Block。
        # 4. 对每个 expert 的输出做 final RMSNorm。
        # 5. 返回每个 expert 的输出，以及可复用的 KV cache。
        #
        # 它不做的事情：
        # - 不负责把图像编码成 tokens；那在 `pi0.py::embed_prefix()` 通过 SigLIP 完成。
        # - 不负责把 action 投影成 tokens；那在 `pi0.py::embed_suffix()` 完成。
        # - 不负责把 suffix_out 变成动作；那在 `pi0.py::action_out_proj` 完成。
        #
        # 所以这个函数可以理解成：
        # “给定已经准备好的 prefix/action tokens，把它们送进多 expert Transformer 里交换信息。”
        # 输入:
        # - embedded: list of expert inputs。
        #   embedded[0] 属于 PaliGemma expert，宽度通常是 2048。
        #   embedded[1] 属于 action expert，宽度通常是 1024。
        # - positions: 当前输入 tokens 的位置索引。
        # - mask: attention mask，进来时是 [batch, query_len, key_len]。
        # - adarms_cond: 每个 expert 的 adaptive RMSNorm 条件。
        # - kv_cache: 推理阶段可选，用于缓存 prefix 的 K/V。
        # - deterministic: 控制 dropout。
        #
        # 输出:
        # - embedded: list of outputs，结构和输入 list 对齐。
        #   pi0.py 中通常解包成 `(prefix_out, suffix_out)`。
        # - kv_cache: 全层 K/V 缓存，sample_actions 里会被后续 denoising step 复用。
        # embedded 的 list 结构就是“哪个 expert 本轮有 token 要跑”：
        #
        # 训练:
        #   embedded = [prefix_tokens, suffix_tokens]
        #
        # 推理填 cache:
        #   embedded = [prefix_tokens, None]
        #
        # 推理 denoise step:
        #   embedded = [None, suffix_tokens]
        #
        # 这种接口设计让 openpi 可以在训练时完整建图，在推理时复用 prefix cache。
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)  # 统一把输入 hidden states 转成主计算精度。
        mask = jnp.asarray(mask)[:, None, :, :]  # attention 实现期望 [b, 1, t, s] 形式的 mask，这里补一个单例 head 维。
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)  # 默认所有 expert 都走普通 RMSNorm 分支。
        # adarms_cond 的 list 长度必须和 configs 对齐。
        # 在 openpi 当前 pi0.5 主路径里，它通常是 [None, time_emb]：
        # 第一个 None 表示 PaliGemma expert 不做 adaptive norm，
        # 第二个 time_emb 表示 action expert 根据 flow timestep 调制。

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)  # 整栈 block 同步推进所有 expert 的 token 流。

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache  # final_norm 也能感知 adarms_cond，所以 pi0.5 的条件会一路传到最后一层。

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        # init 是给 `nnx_bridge.ToNNX(...).lazy_init(...)` 用的初始化辅助函数。
        #
        # 它为什么存在：
        # - Flax Linen 只有在某条代码路径真的执行过时，才会创建那条路径上的参数。
        # - pi0.5 的 adaptive RMSNorm 只有 cond 不为 None 时才会创建 Dense 参数。
        # - 如果初始化时没有故意跑一次 adaRMS 分支，后面加载/训练 pi0.5 参数就会缺参数。
        #
        # 所以这个函数用假输入强行走一遍关键路径：
        # - 先初始化 embedding table。
        # - 再初始化所有 Block、Norm、Attention、FFN、可选 adaRMS 参数。
        #
        # 输入 use_adarms 控制每个 expert 是否需要创建 adaRMS 参数。
        # 输入:
        # - use_adarms: list[bool]，长度和 configs 一致，表示每个 expert 初始化时是否要走 adaptive RMSNorm 分支。
        # 输出:
        # - 无显式输出；副作用是让 Linen 创建所有可能需要的参数。
        #
        # 上游 `Pi0.__init__` 会在 pi0.5 时传 [False, True]：
        # 第 0 个 PaliGemma expert 不需要 adaRMS；第 1 个 action expert 需要 adaRMS。
        # Linen 是惰性创建参数的：某个分支如果没有跑过，对应参数就不会被创建。
        # pi0.5 的 adaptive RMSNorm 只有在 cond 不为 None 时才会创建 Dense 参数，
        # 所以 `pi0.py` 初始化时会传 use_adarms=[False, True]，强制 action expert 的 adaRMS 参数被建出来。
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))  # 先初始化 embedder 参数。
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )  # 再用一轮假输入把 block / norm / adaRMS 相关参数全部惰性初始化出来。


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    # _apply_rope 的作用：
    # 给 attention 的 query/key 注入相对位置信息。
    #
    # 它在系统中的位置：
    # - `Attention.__call__` 在算 attention logits 之前分别对 q 和 k 调用它。
    #
    # 为什么只处理 q/k：
    # - attention 权重来自 q 和 k 的点积。
    # - 只要旋转 q/k，位置关系就会体现在 attention 分数里。
    # - value 负责被加权汇总内容，不需要做 RoPE。
    #
    # 为什么 openpi 推理时尤其需要位置正确：
    # - prefix 先被缓存。
    # - suffix 后面单独前向。
    # - suffix 的 positions 必须接在 prefix 后面，否则它读取 prefix KV cache 时位置关系会错。
    # 输入:
    # - x: query 或 key 张量，形状类似 [batch, seq_len, heads, head_dim]。
    # - positions: 每个 token 的位置 id，形状 [batch, seq_len]。
    # - max_wavelength: RoPE 最大波长，控制最低频位置编码。
    #
    # 输出:
    # - res: 和 x 同形状的旋转后 q/k。
    #
    # 它只作用于 q/k，不作用于 v，因为 RoPE 是通过旋转 query/key 来影响点积注意力分数。
    # RoPE 这里服务的是“统一拼接后的 token 序列位置”。
    # 在训练时，positions 来自 prefix+suffix 的整体 cumsum；
    # 在推理 suffix-only 时，`pi0.py` 会用 prefix 长度作为 offset，让 suffix 位置接在 prefix 后面。
    # 这样 suffix token 在读取 prefix KV cache 时，位置编码仍然和完整序列前向一致。
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)  # 为每一对偶数/奇数通道生成对应频率指数。
    timescale = max_wavelength**freq_exponents  # 得到从短波到长波的一组 RoPE 时间尺度。
    radians = positions[..., None] / timescale[None, None, :]  # 把离散位置 index 映射成每个频率下的转角。
    radians = radians[..., None, :]  # 再插入一个 head 相关广播维，方便和 x 对齐。
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)  # 把最后一维按偶/奇通道一分为二，做复平面旋转。
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)  # 这就是标准 RoPE 旋转公式。
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)  # 最终转回原输入 dtype，避免后续 attention 张量精度不一致。


def _name(name, i):
    # _name 的作用：
    # 给不同 expert 的参数生成不同名字，同时让第 0 个 expert 保留原始 PaliGemma 参数名。
    #
    # 它在系统中的位置：
    # - Attention 的 q/k/v/out 投影会调用它。
    # - Block 里的 pre_attention_norm / pre_ffw_norm / mlp 会调用它。
    # - Module 里的 final_norm 也会调用它。
    #
    # 为什么第 0 个 expert 不加后缀：
    # - 第 0 个 expert 要尽量和原始 PaliGemma checkpoint 的参数路径对齐。
    # - 这样可以无缝加载预训练视觉语言模型权重。
    #
    # 为什么后续 expert 加后缀：
    # - action expert 不能和 PaliGemma expert 共用同一套参数。
    # - 加 `_1` 这样的后缀可以在 checkpoint 里清楚地区分。
    # 输入:
    # - name: 原始参数/子模块名，例如 "attn"、"mlp"、"pre_attention_norm"。
    # - i: expert index。
    # 输出:
    # - 第 0 个 expert 返回原名；后续 expert 返回带后缀的名字。
    #
    # 这个命名规则直接影响 checkpoint 参数路径，也是 JAX -> PyTorch 转换脚本能区分 PaliGemma
    # 和 action expert 参数的基础。
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name  # 第一个 expert 保留原始参数名，方便直接从官方 PaliGemma 权重对齐加载。
    return f"{name}_{i}"  # 后续 expert 追加后缀，比如 attn_1 / mlp_1。


def _gated_residual(x, y, gate):
    # _gated_residual 的作用：
    # 把 attention/FFN 子层输出写回 residual stream。
    #
    # 普通 Transformer:
    # - 没有 gate 时，就是 `x + y`。
    #
    # pi0.5:
    # - adaRMSNorm 会根据 timestep 产生 gate。
    # - 有 gate 时，就是 `x + y * gate`。
    # - 这让模型能根据当前 flow timestep 控制“这一层更新量写回多少”。
    #
    # 它在系统中的位置：
    # - `Block.__call__` 在 attention 后调用一次。
    # - `Block.__call__` 在 FFN 后再调用一次。
    #
    # 它为什么要单独写成函数：
    # - 因为同一套逻辑要同时兼容普通 residual、pi0.5 gated residual、以及某个 expert 为 None 的情况。
    # 输入:
    # - x: residual 主分支输入，形状 [batch, seq_len, width] 或 None。
    # - y: attention/FFN 子层产生的更新量，形状与 x 一致或 None。
    # - gate: adaptive RMSNorm 产生的门控，形状通常是 [batch, 1, width]，或 None。
    # 输出:
    # - 普通 residual 或 gated residual 后的 hidden states。
    # 普通 Transformer residual 是 x + y。
    # pi0.5 的 adaRMSNorm 会额外产生 gate，于是 residual 变成 x + gate * y。
    # 这让 flow timestep 不只是影响 norm 后的特征，还能影响“这一层更新量写入多少”。
    assert (x is None) == (y is None)  # 输入和增量必须同时存在或同时缺失。
    if x is None:
        return None  # 当前 expert 本轮没参与前向，就保持 None。
    if gate is None:
        return x + y  # 普通残差：直接相加。
    return x + y * gate  # gated residual：先用 gate 调制增量，再加回主分支；这是 pi0.5 条件化的一部分。
