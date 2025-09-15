To use the this code, we must install [cleanrl](https://github.com/vwxyzjn/cleanrl) and [Jax](https://github.com/jax-ml/jax) with Python 3.10.18

For the pretrain of the **pretrained_network**, we must use the  code in [ReinFlow](https://github.com/ReinFlow/ReinFlow). Now we first provide two existing pretrained network for environment: Walker2d-v2 and HalfCheetah-v2.

For the fine tuning code on Robomimic and OGbench, use the code in **qc**, which is developed on [Reinforcement Learning with Action Chunking](https://github.com/ColinQiyangLi/qc). Based on the current python env, we still need to install:

`pip install dm_control==1.0.28 ml_collections==1.1.0 ogbench==1.1.2`

Meanwhile, we need to install robomimic through:

[Installation — robomimic 0.5 documentation](https://robomimic.github.io/docs/introduction/installation.html), you may encounter the error like 

`jax.random.key`, to handle this, just:
`pip install diffusers==0.21.4`

Four main algorithms can be implemented as the example:

QC-FQL:
` MUJOCO_GL=egl  python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5`

FQL:
` MUJOCO_GL=egl  python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=1`

Flow-T-SAC:
`MUJOCO_GL=egl python main_action_reg_three_phase.py --run_group=reproduce --agent=agents/acfql_transformer_ablation_online_sac.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5`

Flow-G-SAC:
`MUJOCO_GL=egl python main_action_reg_three_phase.py --run_group=reproduce --agent=agents/acfql_gru_ablation_online_sac.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5`