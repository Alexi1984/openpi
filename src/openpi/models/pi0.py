import logging  # logging: 本文件只在少数地方打印调试信息，这里先拿到 openpi 的统一 logger。

import einops  # einops: 主要用来做张量维度重排和 repeat，方便构造 attention mask / time tokens。
import flax.nnx as nnx  # nnx: Flax 的新模块系统；Pi0 本体就是基于 nnx.Module 实现的。
import flax.nnx.bridge as nnx_bridge  # nnx_bridge: 这里用来把旧的 Flax/Linen 风格模块桥接到 NNX 世界里。
import jax  # jax: 本文件的随机数、while_loop、einsum、数组计算都基于 JAX。
import jax.numpy as jnp  # jnp: JAX 版本的 numpy API，是这里的主要张量运算接口。
from typing_extensions import override  # override: 显式标注这是在覆写基类接口，便于阅读和静态检查。

from openpi.models import model as _model  # _model: 通用模型抽象定义，里面有 BaseModel / Observation / preprocess_observation。
from openpi.models import pi0_config  # pi0_config: Pi0 的配置 dataclass，就在这里定义 action_dim / pi05 开关等。
import openpi.models.gemma as _gemma  # _gemma: 语言主干和 action expert 都复用了 Gemma 模块实现。
import openpi.models.siglip as _siglip  # _siglip: 图像编码器实现；把图像转成视觉 tokens。
from openpi.shared import array_typing as at  # at: 项目自定义的数组类型标注和运行时 typecheck 装饰器。

logger = logging.getLogger("openpi")  # logger: 统一使用 openpi 命名空间下的日志器，便于全项目日志汇总。


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # make_attn_mask:
    # 这个辅助函数的任务是把“哪些 token 有效”和“哪些 token 应该因果遮罩”这两类信息，
    # 合成最终 Transformer 真的能用的三维 attention mask。
    # 这个函数是读懂 prefix / suffix 交互方式的关键。

    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # 先把 mask_ar 广播到和 input_mask 同形状，保证 batch 维对齐。
    cumsum = jnp.cumsum(mask_ar, axis=1)  # cumsum: 用累计和把“块边界/因果边界”编码成单调不减的段编号。
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # 只有“key 所在段编号 <= query 所在段编号”时才允许注意。
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # valid_mask: padding token 之间一律屏蔽，只保留真实输入 token。
    return jnp.logical_and(attn_mask, valid_mask)  # 最终 mask: 同时满足因果约束和非 padding 约束。


@at.typecheck  # typecheck: 运行时帮我们检查输入输出 shape/type 是否符合注解，调试时很有用。
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # 这个函数把一个标量位置/时间 `pos` 编码成标准的 sin-cos embedding。
    # 在 Pi0 里，它不是给“token 位置”做 embedding，而是给 flow matching 的 timestep 做 embedding。

    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")  # sin 和 cos 各占一半维度，所以总维度必须能被 2 整除。

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # fraction: 在 [0, 1] 上均匀取点，用来生成一组从短周期到长周期的频率。
    period = min_period * (max_period / min_period) ** fraction  # period: 对数尺度插值后的周期序列，覆盖 min_period 到 max_period。
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,  # pos: batch 中每个样本自己的时间标量。
        1.0 / period * 2 * jnp.pi,  # 这一项把 period 变成角频率 omega = 2π / period。
        precision=jax.lax.Precision.HIGHEST,  # 这里显式要求高精度 einsum，减少数值误差。
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)  # 最终把 sin 和 cos 特征拼起来。


class Pi0(_model.BaseModel):  # Pi0: 这是 openpi 里 pi0 / pi0.5 这条模型线的核心实现类。
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)  # 先初始化基类记录的几个全局超参数。
        self.pi05 = config.pi05  # self.pi05: 用一个布尔开关区分当前实例到底走 pi0 还是 pi0.5 分支。
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # paligemma_config: 语言/多模态主干的 Gemma 配置。
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # action_expert_config: 动作 expert 那部分的 Gemma 配置。
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # configs: 这里一次构造了两个 Gemma 配置，对应主干和 action expert。
                embed_dtype=config.dtype,  # embed_dtype: embedding 层等使用的数值精度。
                adarms=config.pi05,  # adarms: 只有 pi0.5 才启用 adaRMSNorm 形式的时间条件注入。
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])  # lazy_init: 先按配置初始化参数；pi0.5 时只给第二个 expert 打开 adaRMS。
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # num_classes 这里其实被用成输出宽度，让视觉 token 维度和语言主干宽度对齐。
                variant="So400m/14",  # variant: 指定使用哪一个 SigLIP 视觉骨干。
                pool_type="none",  # pool_type='none': 不做全局池化，因为这里要保留 patch token 序列。
                scan=True,  # scan=True: 通常是为了更省显存/更适合长序列执行。
                dtype_mm=config.dtype,  # dtype_mm: 多模态部分使用的计算精度。
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)  # 用 fake observation 里的任意一张假图像完成视觉编码器的惰性初始化。
        self.PaliGemma = nnx.Dict(llm=llm, img=img)  # 把语言主干和图像编码器打包成一个 Dict，后面统一从 self.PaliGemma 访问。
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # action_in_proj: 把动作从原始 action_dim 投到 action expert 的隐藏维度。
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  # time_mlp_in: pi0.5 中时间条件 MLP 的第一层。
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  # time_mlp_out: pi0.5 中时间条件 MLP 的第二层。
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # state_proj: pi0 中把连续 state 压成一个单独的 state token。
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)  # 把 action token 和 time token 拼接后再降回 expert 宽度。
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  # 上面 MLP 的第二层，进一步融合动作与时间信息。
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)  # action_out_proj: 把 expert hidden state 再投回动作空间维度。

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True  # deterministic: NNX 会在 train/eval 模式切换时自动改这个标志，影响 dropout 等随机层行为。

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # embed_prefix:
        # 把“前缀部分”的输入编码成 Transformer token 序列。
        # 这里的前缀主要包含图像 token 和语言 token，它们代表“条件信息”，不是要被去噪的动作。
        input_mask = []  # input_mask: 记录每个 prefix token 是否有效（不是 padding）。
        ar_mask = []  # ar_mask: 记录 prefix 内部的自回归结构，后面会被 make_attn_mask 转成真正的 attention mask。
        tokens = []  # tokens: 收集各个模态编码后的 token 序列，最后拼接成统一 prefix。
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  # 用 SigLIP 把某一路图像编码成 patch token 序列。

            tokens.append(image_tokens)  # 把这一视角的图像 token 追加到总 token 列表里。
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],  # obs.image_masks[name]: 每张图像只有一个布尔 mask，需要扩展到“每个 patch token 一个 mask”。
                    "b -> b s",
                    s=image_tokens.shape[1],  # s: 当前这路图像被编码成了多少个 patch token。
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]  # False 表示同一块里共享注意力级别；这里让图像 token 之间全互相可见。

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # 只做 embedding，不跑完整 Transformer。
            tokens.append(tokenized_inputs)  # 把语言 token 放到图像 token 后面，构成统一 prefix。
            input_mask.append(obs.tokenized_prompt_mask)  # 语言自己的 padding mask 直接沿用。
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]  # 语言 token 同样作为 prefix 条件，彼此与图像都全可见。
        tokens = jnp.concatenate(tokens, axis=1)  # 把所有前缀 token 沿序列维拼起来，得到 [b, s, emb]。
        input_mask = jnp.concatenate(input_mask, axis=1)  # 把对应的有效位 mask 也沿序列维拼起来。
        ar_mask = jnp.array(ar_mask)  # ar_mask 转成 JAX 数组，后面好喂给 make_attn_mask。
        return tokens, input_mask, ar_mask  # 返回 prefix token、本体 mask、以及用于构造注意力的 autoregressive mask。

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        # embed_suffix:
        # 把“后缀部分”的输入编码成 token 序列。
        # 后缀的核心是 noisy action；在 pi0 里还额外包含一个连续 state token；
        # 在 pi0.5 里 state 已经离散化进语言/token 体系，所以这里不再单独加 state token。
        input_mask = []  # suffix token 的有效位 mask。
        ar_mask = []  # suffix token 的自回归结构定义。
        tokens = []  # suffix token 序列缓冲区。
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]  # 把连续 state 向量投影成 1 个 token；[:, None, :] 是为了显式插入序列维。
            tokens.append(state_token)  # 在 pi0 中，state token 作为 suffix 的第一个 token。
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))  # state token 永远有效，所以 mask 全 1。
            # image/language inputs do not attend to state or actions
            ar_mask += [True]  # True 在这里意味着“从这一 token 开始进入新的因果块”，前面的 prefix 不能反向依赖它。

        action_tokens = self.action_in_proj(noisy_actions)  # 把 noisy action 序列投影到 expert hidden dim，每个时间步对应一个 token。
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)  # timestep 先编码成一个和 action hidden dim 一样宽的时间向量。
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)  # 第一层时间 MLP。
            time_emb = nnx.swish(time_emb)  # swish: 非线性激活，增强时间条件表达能力。
            time_emb = self.time_mlp_out(time_emb)  # 第二层时间 MLP。
            time_emb = nnx.swish(time_emb)  # 再过一次非线性，得到更适合拿去做 adaRMS 条件的向量。
            action_expert_tokens = action_tokens  # pi0.5 不把时间直接拼到 action token 上，而是另走 adaRMS 条件分支。
            adarms_cond = time_emb  # adarms_cond: 供 Gemma/action expert 内部的 adaRMSNorm 使用。
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # 把每个样本的时间向量复制到 action horizon 的每个时间步。
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # 在最后一维拼接 action 信息和 time 信息。
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)  # 第一层融合 MLP，把拼接后的 2*width 压回 width。
            action_time_tokens = nnx.swish(action_time_tokens)  # 非线性激活。
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # 第二层融合 MLP。
            action_expert_tokens = action_time_tokens  # pi0 中真正送给 action expert 的 token 是“动作+时间已混合”的 token。
            adarms_cond = None  # pi0 不使用 adaRMS 条件，因此这里传 None。
        tokens.append(action_expert_tokens)  # 把动作 expert token 序列接到 suffix 里。
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  # 动作 token 序列也都是真实 token，因此 mask 全 1。
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))  # 第一个 action token 开启新的因果块；后续 action token 在同一块中再保持内部因果结构。
        tokens = jnp.concatenate(tokens, axis=1)  # 拼出完整 suffix token 序列。
        input_mask = jnp.concatenate(input_mask, axis=1)  # 拼出 suffix 的有效位 mask。
        ar_mask = jnp.array(ar_mask)  # 转成 JAX 数组以便后续参与 attention mask 构造。
        return tokens, input_mask, ar_mask, adarms_cond  # 最后连同可选的 adaRMS 时间条件一起返回。

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # compute_loss:
        # 训练时的核心入口。这里实现的是 flow matching 风格的损失：
        # 1. 对真实动作 actions 加噪得到 x_t；
        # 2. 让模型预测速度场 v_t；
        # 3. 用 v_t 去拟合真实目标速度 u_t。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)  # 把总随机数拆成三路：观测预处理、噪声采样、时间采样。
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # 统一做图像 resize/增强/补 mask 等预处理。

        batch_shape = actions.shape[:-2]  # batch_shape: 动作最后两维是 [action_horizon, action_dim]，前面都视作 batch 维。
        noise = jax.random.normal(noise_rng, actions.shape)  # noise: 从标准高斯采样，与 actions 同 shape。
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # time: 在 (0,1) 里采样流时间，偏向靠近 1 但避开精确端点。
        time_expanded = time[..., None, None]  # 扩成 [..., 1, 1]，这样能和动作张量广播相乘。
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # x_t: 在 actions 和 noise 之间做线性插值，得到时刻 t 的中间状态。
        u_t = noise - actions  # u_t: 这条直线路径对应的真速度场目标，是训练时要拟合的监督信号。

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # 先把图像/语言条件编码成 prefix。
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)  # 再把 noisy action 和时间条件编码成 suffix。
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # 整条序列的有效位 mask = prefix mask + suffix mask。
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # 整条序列的自回归结构 = prefix 结构 + suffix 结构。
        attn_mask = make_attn_mask(input_mask, ar_mask)  # 真正喂给 Transformer 的 attention mask。
        positions = jnp.cumsum(input_mask, axis=1) - 1  # positions: 只对有效 token 递增计数，padding 位置也会被 mask 掉。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )  # 一次性把 prefix 和 suffix 都前向过去，让模型在条件上下文里预测动作速度。
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # 只取 suffix 最后那段对应 action horizon 的输出，再投回动作空间，得到预测速度场。

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # flow matching MSE：按动作维求均方误差，保留 batch 和 horizon 维。

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # sample_actions:
        # 推理时的核心入口。这里做的是从纯噪声开始，按 learned velocity field 一步步积分到动作空间，
        # 相当于在 ODE / flow matching 的视角下进行去噪采样。
        observation = _model.preprocess_observation(None, observation, train=False)  # 推理时也要做同样的观测预处理，但不做数据增强。
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps  # dt 为负数，因为这里是从 t=1（纯噪声）往 t=0（目标动作）反向积分。
        batch_size = observation.state.shape[0]  # batch_size: 推理时一批 observation 里有多少个样本。
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  # 默认从标准高斯初始化整段动作序列。

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # 先把条件前缀编码出来。
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  # prefix 自己内部的 attention mask。
        positions = jnp.cumsum(prefix_mask, axis=1) - 1  # prefix token 的位置索引。
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)  # 先跑一遍 prefix，把 KV cache 填好，后面每一步复用。

        def step(carry):
            x_t, time = carry  # x_t: 当前时刻的 noisy action；time: 当前积分时间标量。
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)  # 把标量 time 广播到 batch 维，让每个样本共用同一个当前积分时刻。
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # suffix 内部自注意力 mask。
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  # 把 prefix 的有效 token 扩成“每个 suffix query 都能看到的 prefix key mask”。
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)  # 把 prefix 部分和 suffix 部分的可见性拼成完整 mask。
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )  # 这里显式检查 shape，保证我们构造的“query 对 prefix+suffix 的可见性”没有错。
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  # suffix 的位置要从 prefix 长度之后继续编号。

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],  # prefix 已经通过 KV cache 提供，这里不需要重复传 prefix tokens。
                mask=full_attn_mask,  # 只让 suffix query 看见“该看的 prefix + suffix”。
                positions=positions,  # 提供 suffix token 在全序列中的位置编号。
                kv_cache=kv_cache,  # 复用 prefix 的 KV cache，避免每个采样 step 都重算 prefix。
                adarms_cond=[None, adarms_cond],  # prefix 没有时间条件；suffix 这一路可能有 adaRMS 时间条件。
            )
            assert prefix_out is None  # 因为这次只前向 suffix，所以 prefix_out 理论上必须为空。
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # 把 suffix 输出映射回动作速度场预测。

            return x_t + dt * v_t, time + dt  # Euler 积分一步：x_{t+dt} = x_t + dt * v_t，同时更新时间。

        def cond(carry):
            x_t, time = carry  # x_t 在条件函数里没有直接用到，但保持 carry 结构一致更方便 while_loop。
            # robust to floating-point error
            return time >= -dt / 2  # 终止条件写得稍微宽一点，避免浮点误差导致最后一步少走或多走。

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  # 从 t=1 的高斯噪声出发，循环积分到 t≈0。
        return x_0  # 最终得到的 x_0 就是采样出的动作序列。
