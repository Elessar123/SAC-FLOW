from os.path import expanduser
import os

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY

from utils.datasets import Dataset


def is_robomimic_env(env_name):
    """determine if an env is robomimic"""
    if "low_dim" not in env_name:
        return False
    task, dataset_type, hdf5_type = env_name.split("-")
    return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in ("mh", "ph")


low_dim_keys = {"low_dim": ('robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object')}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


def make_env(env_name, seed=0):
    """
    NOTE: should get_dataset() first, so that the metadata is downloaded before creating the environment
    """
    # _download_dataset_and_metadata(env_name)
    dataset_path = _check_dataset_exists(env_name)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
    )
    env = RobomimicLowdimWrapper(env, low_dim_keys=low_dim_keys["low_dim"], max_episode_length=max_episode_length)
    env.seed(seed)
    
    return env

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    task, dataset_type, hdf5_type = env_name.split("-")
    if dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = "low_dim_v15.hdf5"
    download_folder = os.path.join(
        expanduser("~/.robomimic"), 
        task,
        dataset_type,
        file_name
    )
    assert os.path.exists(download_folder)
    
    return download_folder

def get_dataset(env, env_name):
    dataset_path = _check_dataset_exists(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")
    example_action = env.action_space.sample()
    
    # data holder
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    
    # go through and add to the data holder
    for ep in demos:
        a = np.array(rm_dataset["data/{}/actions".format(ep)])
        obs, next_obs = [], []
        for k in low_dim_keys["low_dim"]:
            obs.append(np.array(rm_dataset[f"data/{ep}/obs/{k}"]))
        for k in low_dim_keys["low_dim"]:
            next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]))
        obs = np.concatenate(obs, axis=-1)
        next_obs = np.concatenate(next_obs, axis=-1)
        dones = np.array(rm_dataset["data/{}/dones".format(ep)])
        r = np.array(rm_dataset["data/{}/rewards".format(ep)])
        
        observations.append(obs.astype(np.float32))
        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
    
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )


class RobomimicLowdimWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces - use [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1.)
        high = np.full(env.action_dimension, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self):
        raw_obs = self.env.get_observation()
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""

        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            # always reset to the same state to be compatible with gym
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            # random reset
            self.env.reset()

        return self.get_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(raw_obs)
        else:
            obs = raw_obs

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # print(obs, reward, done, info)
        if reward > 0.:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0

        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
    
    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}


if __name__ == "__main__":
    # for testing 
    env = make_env("lift-mh-low_dim")
    dataset = get_dataset(env, "lift-mh-low_dim")
    print(dataset)
    # transport-mh-low_dim