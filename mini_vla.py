#!/usr/bin/env python3
"""
Minimal Diffusion-based Vision-Language-Action (VLA) Model
==========================================================

A simplified educational implementation of a diffusion policy for robotic manipulation
that combines vision (images), language (text commands), and proprioception (joint positions).

Key Components:
1. Data Collection: Gather demonstrations (images + joint positions + text + actions)
2. Model: Diffusion Transformer that predicts actions from observations
3. Training: Learn from demonstrations using diffusion denoising
4. Inference: Generate actions by denoising from random noise

Architecture:
- Observation Encoder: Processes images (ResNet) + text (CLIP) + joint positions
- Diffusion Transformer: Predicts actions conditioned on observations
- Action Decoder: Converts denoised predictions to robot controls

Usage:
    # Collect data (100 episodes)
    python mini_vla.py --mode collect --num_episodes 100
    
    # Train model (50 epochs)
    python mini_vla.py --mode train --num_epochs 50
    
    # Test model (10 episodes with visualization)
    python mini_vla.py --mode test --num_episodes 10 --visualize
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm

# Import our utilities
from utils.env import FrankaEnv
from utils.data import collect_demonstrations, VLADataset
from utils.model import VLAModel, SimpleObsEncoder
from utils.helpers import normalize_data, denormalize_data, set_seed


class MinimalVLA:
    """Main class orchestrating data collection, training, and inference."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path("data")
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Model hyperparameters
        self.horizon = 16  # Predict 16 steps ahead
        self.n_obs_steps = 2  # Use 2 observation steps for context
        self.n_action_steps = 8  # Execute 8 action steps before replanning
        
        print(f"Using device: {self.device}")
        print(f"Horizon: {self.horizon}, Obs steps: {self.n_obs_steps}, Action steps: {self.n_action_steps}")
    
    def collect_data(self):
        """Collect demonstration trajectories."""
        print(f"\n{'='*60}")
        print(f"COLLECTING {self.args.num_episodes} EPISODES")
        print(f"{'='*60}\n")
        
        self.data_dir.mkdir(exist_ok=True)
        
        collect_demonstrations(
            data_dir=self.data_dir,
            num_episodes=self.args.num_episodes,
            max_steps=200,
            visualize=self.args.visualize
        )
        
        print(f"\n✓ Data collection complete! Files saved to {self.data_dir}/")
    
    def train(self):
        """Train the diffusion VLA model."""
        print(f"\n{'='*60}")
        print(f"TRAINING FOR {self.args.num_epochs} EPOCHS")
        print(f"{'='*60}\n")
        
        # 1. Load dataset
        print("Loading dataset...")
        dataset = VLADataset(
            data_dir=self.data_dir,
            horizon=self.horizon,
            device=self.device
        )
        
        if len(dataset) == 0:
            raise ValueError(f"No data found in {self.data_dir}. Run --mode collect first!")
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        print(f"  Loaded {len(dataset)} sequences from {dataset.num_episodes} episodes")
        print(f"  Image shape: {dataset.image_shape}")
        print(f"  Qpos dim: {dataset.qpos_dim}, Action dim: {dataset.action_dim}")
        
        # 2. Create model
        print("\nCreating model...")
        obs_encoder = SimpleObsEncoder(
            image_shape=dataset.image_shape,
            qpos_dim=dataset.qpos_dim,
            embed_dim=256,
            device=self.device
        )
        
        model = VLAModel(
            obs_encoder=obs_encoder,
            action_dim=dataset.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            embed_dim=256,
            n_heads=4,
            n_layers=4,
            device=self.device
        )
        
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=1e-4
        )
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        
        # 4. Training loop
        print(f"\nStarting training...")
        model.train()
        
        for epoch in range(self.args.num_epochs):
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Unpack batch
                images = batch['image'].to(self.device)  # (B, T, 3, H, W)
                qpos = batch['qpos'].to(self.device)  # (B, T, 7)
                text = batch['text']  # List of strings
                actions = batch['action'].to(self.device)  # (B, T, 8)
                
                B = images.shape[0]
                
                # Normalize (simple min-max to [-1, 1])
                norm_qpos = normalize_data(qpos, dataset.qpos_stats)
                norm_actions = normalize_data(actions, dataset.action_stats)
                norm_images = images / 127.5 - 1.0  # [0,255] -> [-1,1]
                
                # Prepare observations (first n_obs_steps)
                obs_dict = {
                    'image': norm_images[:, :self.n_obs_steps],  # (B, n_obs_steps, 3, H, W)
                    'qpos': norm_qpos[:, :self.n_obs_steps],  # (B, n_obs_steps, 7)
                    'text': text  # List of B strings
                }
                
                # Forward diffusion: add noise to actions
                noise = torch.randn_like(norm_actions)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()
                
                noisy_actions = noise_scheduler.add_noise(norm_actions, noise, timesteps)
                
                # Predict noise
                pred_noise = model(
                    noisy_actions=noisy_actions,
                    timesteps=timesteps,
                    obs_dict=obs_dict
                )
                
                # Compute loss
                loss = F.mse_loss(pred_noise, noise)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Log
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{self.args.num_epochs} - Avg Loss: {avg_loss:.6f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'qpos_stats': dataset.qpos_stats,
                    'action_stats': dataset.action_stats,
                    'image_shape': dataset.image_shape,
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state': model.state_dict(),
            'qpos_stats': dataset.qpos_stats,
            'action_stats': dataset.action_stats,
            'image_shape': dataset.image_shape,
            'config': {
                'action_dim': dataset.action_dim,
                'qpos_dim': dataset.qpos_dim,
                'horizon': self.horizon,
                'n_obs_steps': self.n_obs_steps,
                'n_action_steps': self.n_action_steps,
            }
        }, final_path)
        
        print(f"\n✓ Training complete! Final model saved to {final_path}")
    
    def test(self):
        """Test the trained model in simulation."""
        print(f"\n{'='*60}")
        print(f"TESTING MODEL FOR {self.args.num_episodes} EPISODES")
        print(f"{'='*60}\n")
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / "final_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train first!")
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model
        config = checkpoint['config']
        dataset_stats = {
            'qpos': checkpoint['qpos_stats'],
            'action': checkpoint['action_stats'],
            'image_shape': checkpoint['image_shape']
        }
        
        obs_encoder = SimpleObsEncoder(
            image_shape=checkpoint['image_shape'],
            qpos_dim=config['qpos_dim'],
            embed_dim=256,
            device=self.device
        )
        
        model = VLAModel(
            obs_encoder=obs_encoder,
            action_dim=config['action_dim'],
            horizon=config['horizon'],
            n_obs_steps=config['n_obs_steps'],
            embed_dim=256,
            n_heads=4,
            n_layers=4,
            device=self.device
        )
        
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        print("✓ Model loaded successfully")
        
        # Create environment
        env = FrankaEnv(visualize=self.args.visualize)
        
        # Create noise scheduler for sampling
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        noise_scheduler.set_timesteps(20)  # Use 20 denoising steps for inference
        
        # Test episodes
        successes = []
        
        for ep in range(self.args.num_episodes):
            print(f"\n--- Episode {ep+1}/{self.args.num_episodes} ---")
            
            # Reset environment and get initial observation
            obs = env.reset(task="pick_place")
            
            # Observation buffers (for history)
            image_history = []
            qpos_history = []
            
            step_count = 0
            max_steps = 200
            
            while step_count < max_steps:
                # Get current observation
                image = obs['image']  # (H, W, 3) uint8
                qpos = obs['qpos']  # (7,) float32
                text = obs.get('text', 'red block')
                
                # Update history
                image_history.append(image)
                qpos_history.append(qpos)
                
                # Keep only last n_obs_steps
                if len(image_history) > self.n_obs_steps:
                    image_history = image_history[-self.n_obs_steps:]
                    qpos_history = qpos_history[-self.n_obs_steps:]
                
                # Pad if needed
                while len(image_history) < self.n_obs_steps:
                    image_history.insert(0, image)
                    qpos_history.insert(0, qpos)
                
                # Prepare observation tensors
                images_np = np.stack(image_history)  # (n_obs_steps, H, W, 3)
                qpos_np = np.stack(qpos_history)  # (n_obs_steps, 7)
                
                # Convert to tensors and normalize
                images_t = torch.from_numpy(images_np).float().to(self.device)
                images_t = images_t.permute(0, 3, 1, 2)  # (n_obs_steps, 3, H, W)
                images_t = images_t / 127.5 - 1.0  # Normalize to [-1, 1]
                
                qpos_t = torch.from_numpy(qpos_np).float().to(self.device)
                qpos_t = normalize_data(qpos_t, dataset_stats['qpos'])
                
                # Add batch dimension
                images_t = images_t.unsqueeze(0)  # (1, n_obs_steps, 3, H, W)
                qpos_t = qpos_t.unsqueeze(0)  # (1, n_obs_steps, 7)
                
                obs_dict = {
                    'image': images_t,
                    'qpos': qpos_t,
                    'text': [text]
                }
                
                # Sample action using diffusion denoising
                with torch.no_grad():
                    # Start from random noise
                    action_shape = (1, self.horizon, config['action_dim'])
                    noisy_actions = torch.randn(action_shape, device=self.device)
                    
                    # Denoise step by step
                    for t in noise_scheduler.timesteps:
                        noise_pred = model(
                            noisy_actions=noisy_actions,
                            timesteps=t.unsqueeze(0),
                            obs_dict=obs_dict
                        )
                        noisy_actions = noise_scheduler.step(
                            noise_pred, t, noisy_actions
                        ).prev_sample
                    
                    # Unnormalize actions
                    sampled_actions = denormalize_data(
                        noisy_actions, dataset_stats['action']
                    )
                    
                    # Extract action chunk (first n_action_steps)
                    action_chunk = sampled_actions[0, :self.n_action_steps].cpu().numpy()
                
                # Execute actions
                for action in action_chunk:
                    obs, reward, done, info = env.step(action)
                    step_count += 1
                    
                    if done or step_count >= max_steps:
                        break
                
                if done or step_count >= max_steps:
                    break
            
            success = info.get('success', False)
            successes.append(success)
            print(f"  Steps: {step_count}, Success: {success}")
        
        # Report results
        success_rate = np.mean(successes)
        print(f"\n{'='*60}")
        print(f"RESULTS: {np.sum(successes)}/{len(successes)} successes ({success_rate*100:.1f}%)")
        print(f"{'='*60}")
        
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Minimal Diffusion VLA")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "train", "test"],
        required=True,
        help="Mode: collect data, train model, or test model"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes (for collect/test mode)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable MuJoCo visualization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create VLA instance
    vla = MinimalVLA(args)
    
    # Run selected mode
    if args.mode == "collect":
        vla.collect_data()
    elif args.mode == "train":
        vla.train()
    elif args.mode == "test":
        vla.test()


if __name__ == "__main__":
    main()