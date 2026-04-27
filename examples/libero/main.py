import collections  # collections.deque: 用作“动作计划队列”，缓存服务端一次返回的一小段 action chunk。
import dataclasses  # dataclass: 用来把 CLI 参数定义成结构化配置，便于 tyro 自动生成命令行接口。
import logging  # logging: 统一打印 rollout 过程、成功率、异常等运行信息。
import math  # math: 这里主要用于四元数 -> axis-angle 的旋转转换。
import pathlib  # pathlib: 用于跨平台地处理视频输出路径、BDDL 文件路径等。

import imageio  # imageio: 把每步保存下来的图像序列写成 mp4 回放视频。
from libero.libero import benchmark  # benchmark: LIBERO 提供的任务基准入口，用来获取 task suite 工厂。
from libero.libero import get_libero_path  # get_libero_path: 定位 LIBERO 自带资源目录（例如 bddl_files）。
from libero.libero.envs import OffScreenRenderEnv  # OffScreenRenderEnv: 不依赖可视窗口、直接离屏渲染的仿真环境。
import numpy as np  # numpy: 图像处理、状态拼接、随机种子等都依赖它。
from openpi_client import image_tools  # image_tools: openpi-client 里的图像预处理工具，和服务端训练预处理保持一致。
from openpi_client import websocket_client_policy as _websocket_client_policy  # WebSocket 客户端：把观测发给策略服务端，再收回动作。
import tqdm  # tqdm: 在评测多任务、多 episode 时显示进度条。
import tyro  # tyro: 根据 dataclass 自动生成命令行参数解析。

# LIBERO_DUMMY_ACTION:
# 在 episode 的最开始几个 step 里，策略还不会真正接管控制，而是先发送一段“静止动作”。
# 这样做是因为仿真器刚 reset 时，物体常常还在掉落 / 稳定，太早执行策略会让初始观测不稳定。
# 这里 7 维动作通常对应：6 维末端执行器控制 + 1 维夹爪控制。
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# LIBERO_ENV_RESOLUTION:
# 这里的环境渲染分辨率故意设成和训练数据生成时一致的 256。
# 后面虽然还会 resize 到 224 发给模型，但先用训练时一致的 render resolution，
# 可以尽量减少“训练图像分布”和“评测图像分布”的偏差。
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # eval_libero: 这是这个文件真正的主函数。
    # 它负责整条“LIBERO 推理评测链路”：
    # 1. 建立 task suite 与 environment；
    # 2. 启动 WebSocket client 连接策略服务端；
    # 3. 把仿真观测整理成服务端所需格式；
    # 4. 接收 action chunk 并以 receding horizon 的方式执行；
    # 5. 统计成功率并保存回放视频。

    # Set random seed
    # 这里只给 numpy 设种子，主要影响某些 numpy 侧随机过程以及和 seed 相关的环境初始化一致性。
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()  # benchmark_dict: 名称 -> task suite 构造器 的映射表。
    task_suite = benchmark_dict[args.task_suite_name]()  # task_suite: 当前选中的任务集合实例，例如 libero_spatial。
    num_tasks_in_suite = task_suite.n_tasks  # n_tasks: 这个 task suite 里一共有多少个任务要逐个评测。
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)  # 确保回放视频输出目录存在。

    # max_steps:
    # 给不同 LIBERO suite 设不同的 rollout 上限，经验上要略大于训练 demo 的最长长度。
    # 这样既能让策略有足够时间完成任务，又避免失败 episode 无限拖长。
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # WebsocketClientPolicy:
    # 这是 openpi 推理实现里非常关键的一层“客户端代理”。
    # 它的 infer(obs) 并不在本地直接跑模型，而是：
    # 1. 把 obs 打包成 msgpack；
    # 2. 通过 websocket 发给 scripts/serve_policy.py 启动的服务端；
    # 3. 再接收服务端回传的动作结果。
    # 所以这个文件本质上是“LIBERO 仿真客户端”，真正的模型推理在服务端那一侧完成。
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0  # 全局统计量：跨任务累计 episode 数和成功数。
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)  # task: 当前任务对象，里面含语言描述、BDDL 文件名、problem_folder 等信息。

        # Get default LIBERO initial states
        # initial_states: LIBERO 为当前任务预先提供的一组初始状态，用于可复现实验评测。
        # 后面的 episode_idx 会直接索引这里，确保每次 rollout 都从 benchmark 定义好的初始状态开始。
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        # _get_libero_env 会把 task 里的 BDDL 信息解析成真正的 MuJoCo 仿真环境。
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0  # 当前任务内部的成功统计，用于看单任务 success rate。
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()  # reset: 先把环境清到初始状态，再单独用 set_init_state 注入 benchmark 提供的确定性初始状态。
            action_plan = collections.deque()  # action_plan: 当前这轮 rollout 还没执行完的 action chunk 队列。

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])  # obs: 设置完任务指定初始状态后返回的第一帧观测。

            # Setup
            t = 0  # t: 当前 episode 的环境步数计数器。
            replay_images = []  # replay_images: 存每一步预处理后的主视角图像，后面写成回放视频。

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    # 这段“等待稳定”逻辑很重要：
                    # 如果仿真 reset 后物体还在下落，策略看到的第一批图像会和训练分布严重不一致。
                    # 因此先执行 num_steps_wait 次 dummy action，让环境稳定后再真正开始感知与推理。
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    # 这里把图像做 [::-1, ::-1] 的双轴翻转，相当于旋转 180 度。
                    # 原因不是“图像增强”，而是为了严格对齐训练数据预处理方向。
                    # 这一点如果漏掉，推理端看到的视觉输入分布会和训练端不一致。
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # resize_with_pad + convert_to_uint8:
                    # 这两步对应 openpi_client/image_tools.py 里的通用图像预处理：
                    # 1. resize_with_pad: 缩放到固定输入大小，同时保持长宽比、不直接拉伸变形；
                    # 2. convert_to_uint8: 把浮点图像压成 uint8，减小 websocket 传输体积，也匹配常见视觉输入格式。
                    # resize_size 默认是 224，这和 openpi/libero policy 期望的输入尺寸一致。
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    # 这里保存的是“发给模型之前的图像”，不是原始仿真图像。
                    # 所以后面回放视频更接近“模型实际看到的内容”。
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # 这里体现的是 receding horizon / chunked control 思路：
                        # 服务端一次会预测一小段未来动作，但客户端不会把整段全执行完才重规划，
                        # 而是只取前 replan_steps 步执行，然后重新观测、重新请求新动作。
                        # 这样可以让策略在闭环中更频繁地根据最新观测纠偏。

                        # Prepare observations dict
                        # element: 这是发给服务端 policy 的原始输入字典。
                        # 字段名必须和 LIBERO 路线的输入适配器对齐。
                        # 具体来说，服务端在 LIBERO 环境下会走 src/openpi/policies/libero_policy.py，
                        # 那里明确要求这些 key：
                        # - observation/image
                        # - observation/wrist_image
                        # - observation/state
                        # - prompt
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    # 训练侧的状态不是直接吃 quaternion，而是吃 axis-angle 形式的姿态表示。
                                    # 所以这里先把末端四元数转换成 3 维 axis-angle，再拼到 state 里。
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            # prompt: 语言任务描述，会被服务端 policy 当作高层任务指令输入。
                            # 对 LIBERO 任务来说，这通常就是 task.language。
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        # client.infer(element) 会把 element 通过 websocket 发送到策略服务端。
                        # 服务端由 scripts/serve_policy.py 启动，内部会创建 LIBERO 对应的 trained policy，
                        # 然后把输入跑过 transforms、模型前向、输出反归一化等完整推理流程。
                        # 返回值里拿到的 "actions" 就是模型预测的一整个 action chunk。
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."

                        # 这里只把 chunk 的前 replan_steps 步塞进 action_plan，
                        # 等执行完这几步就会重新请求一次新 chunk。
                        # 这是当前文件里最关键的“推理控制策略”之一。
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()  # 每个环境 step 只弹出并执行 1 个动作，实现闭环逐步推进。

                    # Execute action in environment
                    # env.step 期待的是 Python list，这里把 numpy action 转成 list 再喂给 LIBERO 环境。
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        # done 为 True 表示当前任务在 LIBERO 环境定义下已经成功完成。
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1  # 只有真正执行了策略动作，才把有效控制步数加 1。

                except Exception as e:
                    # 这里不让单次 rollout 的异常直接炸掉整个评测流程，
                    # 而是记录错误后结束当前 episode，继续后面的任务 / 试验。
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"  # 用 success/failure 区分输出文件名，便于后处理筛查。
            task_segment = task_description.replace(" ", "_")  # 文件名里不用空格，避免路径处理不便。
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language  # task.language: LIBERO 任务自带的自然语言描述，也是后面 prompt 的来源。
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # task_bddl_file: 当前任务对应的 BDDL 场景描述文件。
    # BDDL 里定义了场景物体、关系约束、任务目标等，是构造仿真环境的重要依据。
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)  # OffScreenRenderEnv: 不弹 GUI，只返回渲染图像，适合批量评测和远程运行。
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # _quat2axisangle:
    # LIBERO 观测里末端姿态是 quaternion，
    # 但 openpi 这条 LIBERO 推理线给模型喂的是：位置(3) + 轴角(3) + gripper(若干维) 这种状态向量。
    # 所以这里做的是“环境原始格式 -> 模型训练格式”的一个关键状态表示转换。

    # clip quaternion
    # 先把 w 分量裁到 [-1, 1]，避免后面 acos 因数值误差出现无效输入。
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        # 如果旋转角非常接近 0，那么 axis-angle 的方向实际上不重要，直接返回零向量即可。
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # tyro.cli(eval_libero): 让这个文件既能作为普通 Python 脚本运行，
    # 又能自动把 Args dataclass 暴露成命令行参数。
    # 因此 examples/libero/main.py 本质上就是“LIBERO 推理客户端入口”。
    tyro.cli(eval_libero)
