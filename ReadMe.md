# SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via velocity-reparameterized sequential modeling

## Installation
```bash
conda create -n sacflow python==3.10
pip install requirements.txt
```
Then install third party dependencies:
1. First install [cleanrl](https://github.com/vwxyzjn/cleanrl) and [Jax](https://github.com/jax-ml/jax)
2. Install simulation environments:
```bash
pip install dm_control==1.0.28 ml_collections==1.1.0 ogbench==1.1.2
```

Meanwhile, we need to install robomimic through:
[Installation — robomimic 0.5 documentation](https://robomimic.github.io/docs/introduction/installation.html), you may encounter the error like 
`jax.random.key`, to handle this, just: `pip install diffusers==0.21.4`

## Get Start

Run SAC Flow-T:
``` bash
MUJOCO_GL=egl python main_action_reg_three_phase.py --run_group=reproduce --agent=agents/acfql_transformer_ablation_online_sac.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5
```

Run SAC Flow-G:
```bash
MUJOCO_GL=egl python main_action_reg_three_phase.py --run_group=reproduce --agent=agents/acfql_gru_ablation_online_sac.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5
```

Run QC-FQL:
```bash
MUJOCO_GL=egl  python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=5
```

Run FQL:
```bash
MUJOCO_GL=egl  python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task4-v0 --sparse=False --horizon_length=1
```
