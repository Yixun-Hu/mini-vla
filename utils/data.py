#!/usr/bin/env python3
"""
Data Collection and Dataset for VLA Training

Provides:
1. collect_demonstrations(): Collect trajectories using simple scripted policy
2. VLADataset: PyTorch dataset for loading and batching trajectories
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from utils.env import FrankaEnv


def simple_controller(env, target_cube_id):
    """
    Simple proportional controller to move above target cube.
    
    Returns action to move end-effector above the target cube.
    """
    # Get current positions
    ee_pos = env.data.xpos[env.ee_body_id]
    cube_pos = env.data.xpos[target_cube_id]
    
    # Target: 0.2m above cube
    target_pos = cube_pos + np.array([0, 0, 0.2])
    
    # Simple P controller
    pos_error = target_pos - ee_pos
    
    # Get current joint positions
    current_qpos = env.data.qpos[:7].copy()
    
    # Compute simple Jacobian-like update (very simplified)
    # In real implementation, use proper inverse kinematics
    # Here we just move joints proportionally to error
    K = 3.0  # Gain
    joint_delta = np.random.randn(7) * 0.1  # Add some noise for diversity
    joint_delta[:3] += K * pos_error  # Rough position control
    
    # Clip to reasonable range
    joint_delta = np.clip(joint_delta, -0.2, 0.2)
    
    # Compute target joint positions
    target_qpos = current_qpos + joint_delta
    
    # Create action (7 arm joints + 1 gripper)
    action = np.zeros(8)
    action[:7] = target_qpos
    action[7] = 255.0  # Open gripper
    
    return action


def collect_demonstrations(
    data_dir: Path,
    num_episodes: int = 100,
    max_steps: int = 200,
    visualize: bool = False
):
    """
    Collect demonstration trajectories using simple scripted policy.
    
    Saves each episode as:
        data/episode_XXX.npz with keys:
            - images: (T, H, W, 3) uint8
            - qpos: (T, 7) float32
            - actions: (T, 8) float32
            - text: str
            - success: bool
    """
    data_dir.mkdir(exist_ok=True)
    env = FrankaEnv(visualize=visualize, max_steps=max_steps)
    
    successes = []
    
    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Alternate between red and blue targets
        target = "red" if ep < num_episodes // 2 else "blue"
        target_text = f"{target} block"
        
        # Determine target cube ID
        target_cube_id = env.red_cube_id if target == "red" else env.blue_cube_id
        
        # Reset environment
        obs = env.reset(target=target)
        
        # Episode buffers
        images = []
        qpos_list = []
        actions = []
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Record observation
            images.append(obs['image'])
            qpos_list.append(obs['qpos'])
            
            # Get action from controller
            action = simple_controller(env, target_cube_id)
            actions.append(action)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            step_count += 1
        
        success = info['success']
        successes.append(success)
        
        # Save episode
        episode_data = {
            'images': np.array(images),  # (T, H, W, 3)
            'qpos': np.array(qpos_list),  # (T, 7)
            'actions': np.array(actions),  # (T, 8)
            'text': target_text,  # str
            'success': success  # bool
        }
        
        save_path = data_dir / f"episode_{ep:03d}.npz"
        np.savez_compressed(save_path, **episode_data)
    
    env.close()
    
    success_rate = np.mean(successes)
    print(f"\nCollected {num_episodes} episodes")
    print(f"Success rate: {success_rate*100:.1f}% ({np.sum(successes)}/{num_episodes})")


class VLADataset(Dataset):
    """
    PyTorch dataset for loading VLA demonstration trajectories.
    
    Returns dict with:
        - image: (T, 3, H, W) float32 in [0, 255]
        - qpos: (T, 7) float32
        - action: (T, 8) float32
        - text: str
    """
    
    def __init__(self, data_dir: Path, horizon: int = 16, device='cpu'):
        self.data_dir = data_dir
        self.horizon = horizon
        self.device = device
        
        # Load all episodes
        self.episode_files = sorted(data_dir.glob("episode_*.npz"))
        self.num_episodes = len(self.episode_files)
        
        if self.num_episodes == 0:
            print(f"Warning: No episodes found in {data_dir}")
            self.sequences = []
            return
        
        # Load first episode to get dimensions
        first_ep = np.load(self.episode_files[0])
        self.image_shape = first_ep['images'].shape[1:]  # (H, W, 3)
        self.qpos_dim = first_ep['qpos'].shape[1]  # 7
        self.action_dim = first_ep['actions'].shape[1]  # 8
        
        # Create sequences by sliding window
        self.sequences = []
        all_qpos = []
        all_actions = []
        
        print(f"Creating sequences (horizon={horizon})...")
        for ep_file in tqdm(self.episode_files):
            ep_data = np.load(ep_file)
            images = ep_data['images']  # (T, H, W, 3)
            qpos = ep_data['qpos']  # (T, 7)
            actions = ep_data['actions']  # (T, 8)
            text = str(ep_data['text'])
            
            T = len(images)
            
            # Collect stats
            all_qpos.append(qpos)
            all_actions.append(actions)
            
            # Create overlapping sequences
            for start_idx in range(T - horizon + 1):
                end_idx = start_idx + horizon
                
                seq = {
                    'image': images[start_idx:end_idx],  # (horizon, H, W, 3)
                    'qpos': qpos[start_idx:end_idx],  # (horizon, 7)
                    'action': actions[start_idx:end_idx],  # (horizon, 8)
                    'text': text
                }
                self.sequences.append(seq)
        
        # Compute statistics for normalization
        all_qpos = np.concatenate(all_qpos, axis=0)  # (N, 7)
        all_actions = np.concatenate(all_actions, axis=0)  # (N, 8)
        
        self.qpos_stats = {
            'min': torch.tensor(all_qpos.min(axis=0), dtype=torch.float32),
            'max': torch.tensor(all_qpos.max(axis=0), dtype=torch.float32),
        }
        
        self.action_stats = {
            'min': torch.tensor(all_actions.min(axis=0), dtype=torch.float32),
            'max': torch.tensor(all_actions.max(axis=0), dtype=torch.float32),
        }
        
        print(f"Created {len(self.sequences)} sequences from {self.num_episodes} episodes")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Return a sequence with horizon timesteps."""
        seq = self.sequences[idx]
        
        # Convert to tensors
        # Image: (T, H, W, 3) -> (T, 3, H, W)
        image = torch.from_numpy(seq['image']).float()
        image = image.permute(0, 3, 1, 2)  # Move channel to dim 1
        
        qpos = torch.from_numpy(seq['qpos']).float()
        action = torch.from_numpy(seq['action']).float()
        text = seq['text']
        
        return {
            'image': image,  # (T, 3, H, W)
            'qpos': qpos,  # (T, 7)
            'action': action,  # (T, 8)
            'text': text  # str
        }