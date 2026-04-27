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
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # dtype: 记住输入原始精度，最后要把输出转回这个 dtype。
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # var: RMSNorm 只按最后一维统计均方值，这里用 float32 保证数值更稳。
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # 先完成标准 RMS 归一化，不引入 mean-centering。
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
        return normed_inputs.astype(dtype), gate  # gate 会在残差连接阶段使用，不是这里立即乘上去。


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int  # vocab_size: 离散 token 词表大小。
    embed_dim: int  # embed_dim: 每个 token 被映射到的隐藏维度。

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )  # input_embedding_table: 共享词嵌入矩阵；这里只给第一个 expert 建立嵌入表。

    def encode(self, x):
        x = self.input_embedding_table[(x,)]  # 根据 token id 查表，得到 [b, t, d] 嵌入。
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 按常见 Transformer 做法乘 sqrt(d_model)，保持尺度一致。
        return x  # 返回离散 token 的连续表示。

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)  # 用 embedding matrix 转置做 tied decoding。


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]  # configs: 一个 expert 一个配置；这里要求所有 expert 的注意力头结构兼容。

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        # 这 3 个断言的意义很重要：
        # 因为后面要把多个 expert 的 q/k/v 在序列维直接拼起来统一做 self-attention，
        # 所以头数和 head_dim 必须一致，否则没法共享同一套注意力计算。

        dtype = next(x.dtype for x in xs if x is not None)  # dtype: 拿第一个真实输入 expert 的 dtype 作为当前注意力计算的目标精度。

        qkvs = []  # qkvs: 暂存每个 expert 各自算出来的 (q, k, v)，后面会在序列维拼接。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue  # x is None: 说明这一路 expert 这次不参与前向，比如 prefix-only / suffix-only 模式。
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
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )  # kv_einsum: key/value 头数更少时，共享同一组 kv 头来服务多组 query 头。
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))  # 最后无论哪条分支，都整理成 (q, k, v) 三元组。

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))  # 在序列维拼接各 expert token，这样后面只做一次统一 attention。

        q = _apply_rope(q, positions=positions)  # q 使用 RoPE 注入位置信息。
        q *= self.configs[0].head_dim ** -0.5  # 缩放 query，避免点积随 head_dim 变大而数值过大。

        k = _apply_rope(k, positions=positions)  # k 也必须用同样的 RoPE，才能形成相对位置编码效果。

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache  # kv_cache: 推理时 prefix 先跑过一次后缓存下来的 key/value。
            k = jnp.concatenate([cache_k, k], axis=1)  # 把旧缓存和这一步新 suffix 的 k 拼起来，形成完整上下文。
            v = jnp.concatenate([cache_v, v], axis=1)  # v 同理。

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)  # G: 表示“每个 kv 头服务多少个 query 头”。
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)  # 注意力分数先升到 float32，更稳。

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # 用非常小的负数把不可见位置压到 softmax 后接近 0。
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)  # attn_mask 为 False 的地方全部屏蔽。

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)  # softmax 后再转回原始精度，节省显存。

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)  # 用注意力权重对 value 做加权求和。
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")  # 把分组头再并回标准多头表示。

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

        return out, (k, v)  # 同时返回更新后的 kv，供推理缓存继续复用。


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int  # features: 输入输出维度，也就是 residual stream 宽度。
    hidden_dim: int  # hidden_dim: FFN 中间层维度，通常比 features 大很多。

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # dtype: 保持和外面 residual 一致，避免精度混乱。
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)  # w_gating[0] 和 w_gating[1] 对应 gated-GELU FFN 的两条分支。
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)  # 一条分支过非线性，提供门控。

        ff1 = jnp.dot(x, w_gating[1])  # 另一条分支保持线性激活。
        activations = gate_value * ff1  # 两条分支逐元素相乘，就是 Gemma/GLU 风格 FFN。

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

    configs: tuple[Config, ...]  # configs: 每个 expert 一份配置，但所有 expert 在同一 block 深度上同步推进。

    dropout: float = 0.0  # dropout: 训练时可选的随机失活概率。
    dropout_bdims: tuple[int, ...] = ()  # dropout_bdims: 指定 dropout 共享哪些 batch 维。

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)  # 给激活加 sharding 约束，帮助大模型并行训练。
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x  # 无 dropout 时退化成恒等函数。

        attn = Attention(configs=self.configs, name="attn")  # 所有 expert 共享一套“统一 attention 计算框架”。

        pre_attn = []  # pre_attn: attention 前归一化后的输入。
        gates = []  # gates: 只有 adaRMS 分支会返回 gate，普通 RMSNorm 返回 None。
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901  # pi0.5 就是在这里把 timestep 条件打进 attention 前归一化。
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)  # 真正做 self-attention，并顺便更新 kv cache。
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)  # 训练时可选 dropout。
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]  # attention 残差；若有 gate，则变成条件门控残差。
        xs = sharding.activation_sharding_constraint(xs)

        out = []  # out: FFN 分支的输出。
        gates = []  # 重新收集 FFN 前归一化返回的 gate。
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

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]  # FFN 残差也可能带条件 gate。
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache  # block 输出新的 hidden states，以及当前层更新后的缓存。


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]  # KVCache: (all layer keys, all layer values) 的缓存结构。


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # configs: 每个 expert 一份 Config。当前 openpi 实际上常用 2 个 expert：PaliGemma + action expert。
    embed_dtype: str  # embed_dtype: token embedding / hidden states 的主计算精度。

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False  # adarms: 这里只是配置位，真正是否传 cond 由上层调用时的 adarms_cond 决定。

    def setup(self):
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
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
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
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)  # 统一把输入 hidden states 转成主计算精度。
        mask = jnp.asarray(mask)[:, None, :, :]  # attention 实现期望 [b, 1, t, s] 形式的 mask，这里补一个单例 head 维。
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)  # 默认所有 expert 都走普通 RMSNorm 分支。

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)  # 整栈 block 同步推进所有 expert 的 token 流。

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache  # final_norm 也能感知 adarms_cond，所以 pi0.5 的条件会一路传到最后一层。

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))  # 先初始化 embedder 参数。
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )  # 再用一轮假输入把 block / norm / adaRMS 相关参数全部惰性初始化出来。


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
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
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name  # 第一个 expert 保留原始参数名，方便直接从官方 PaliGemma 权重对齐加载。
    return f"{name}_{i}"  # 后续 expert 追加后缀，比如 attn_1 / mlp_1。


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)  # 输入和增量必须同时存在或同时缺失。
    if x is None:
        return None  # 当前 expert 本轮没参与前向，就保持 None。
    if gate is None:
        return x + y  # 普通残差：直接相加。
    return x + y * gate  # gated residual：先用 gate 调制增量，再加回主分支；这是 pi0.5 条件化的一部分。
