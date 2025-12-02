#!/usr/bin/env python3
"""
Model Components for Diffusion VLA

Components:
1. SimpleObsEncoder: Encodes images (ResNet) + text (CLIP) + qpos
2. DiffusionTransformer: Transformer that predicts noise for diffusion
3. VLAModel: Full model combining encoder + transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import torchvision.models as models


class SimpleObsEncoder(nn.Module):
    """
    Simplified observation encoder that processes:
    - Images via pretrained ResNet18
    - Text via CLIP (optional, can be identity for simplicity)
    - Qpos via linear layer
    
    Returns a fixed-size embedding for each timestep.
    """
    
    def __init__(
        self,
        image_shape=(240, 320, 3),  # (H, W, C)
        qpos_dim=7,
        embed_dim=256,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        
        # Image encoder: pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj = nn.Linear(512, embed_dim)  # ResNet18 output is 512
        
        # Text encoder: for simplicity, use a learned embedding per unique text
        # In full version, this would be CLIP
        self.text_embeddings = nn.Embedding(10, embed_dim)  # Support 10 different texts
        self.text_to_id = {}  # Map text to ID
        
        # Qpos encoder
        self.qpos_proj = nn.Linear(qpos_dim, embed_dim)
        
        # Final projection to combine all modalities
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)
    
    def forward(self, obs_dict: Dict) -> torch.Tensor:
        """
        Args:
            obs_dict with:
                - image: (B, T, 3, H, W) or (B, 3, H, W)
                - qpos: (B, T, 7) or (B, 7)
                - text: List of B strings
        
        Returns:
            embeddings: (B, T, embed_dim) or (B, embed_dim)
        """
        images = obs_dict['image']
        qpos = obs_dict['qpos']
        texts = obs_dict['text']
        
        # Handle both batched time and single timestep
        has_time = (images.dim() == 5)  # (B, T, C, H, W)
        
        if has_time:
            B, T = images.shape[:2]
            # Reshape to (B*T, C, H, W) for batch processing
            images = images.reshape(B * T, *images.shape[2:])
            qpos = qpos.reshape(B * T, -1)
        else:
            B = images.shape[0]
            T = 1
        
        # Encode images
        # Resize to 224x224 for ResNet (expects this size)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images / 255.0 - mean) / std
        
        with torch.no_grad():
            image_features = self.image_encoder(images)  # (B*T, 512, 1, 1)
        image_features = image_features.squeeze(-1).squeeze(-1)  # (B*T, 512)
        image_emb = self.image_proj(image_features)  # (B*T, embed_dim)
        
        # Encode text (simple learned embeddings)
        text_ids = []
        for text in texts:
            if text not in self.text_to_id:
                self.text_to_id[text] = len(self.text_to_id) % 10
            text_ids.append(self.text_to_id[text])
        
        text_ids = torch.tensor(text_ids, device=self.device)  # (B,)
        if has_time:
            # Repeat for each timestep
            text_ids = text_ids.unsqueeze(1).expand(B, T).reshape(B * T)
        text_emb = self.text_embeddings(text_ids)  # (B*T, embed_dim)
        
        # Encode qpos
        qpos_emb = self.qpos_proj(qpos)  # (B*T, embed_dim)
        
        # Fuse all modalities
        combined = torch.cat([image_emb, text_emb, qpos_emb], dim=-1)  # (B*T, 3*embed_dim)
        fused = self.fusion(combined)  # (B*T, embed_dim)
        
        # Reshape back if needed
        if has_time:
            fused = fused.reshape(B, T, self.embed_dim)
        
        return fused


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionTransformer(nn.Module):
    """
    Simplified Transformer for diffusion-based action prediction.
    
    Takes:
    - Noisy actions: (B, T, action_dim)
    - Timestep: (B,) diffusion timestep
    - Condition: (B, T_obs, embed_dim) observation embeddings
    
    Returns:
    - Noise prediction: (B, T, action_dim)
    """
    
    def __init__(
        self,
        action_dim=8,
        horizon=16,
        embed_dim=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        # Action input projection
        self.action_emb = nn.Linear(action_dim, embed_dim)
        
        # Positional embedding for actions
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, embed_dim))
        
        # Transformer decoder (attends to observation condition)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, noisy_actions, timesteps, condition):
        """
        Args:
            noisy_actions: (B, T, action_dim) noisy action sequence
            timesteps: (B,) diffusion timestep
            condition: (B, T_obs, embed_dim) observation embeddings
        
        Returns:
            noise_pred: (B, T, action_dim) predicted noise
        """
        B, T, _ = noisy_actions.shape
        
        # Embed timestep
        time_emb = self.time_emb(timesteps)  # (B, embed_dim)
        time_emb = time_emb.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Add time embedding to condition
        cond_with_time = torch.cat([time_emb, condition], dim=1)  # (B, 1+T_obs, embed_dim)
        
        # Embed noisy actions
        action_emb = self.action_emb(noisy_actions)  # (B, T, embed_dim)
        action_emb = action_emb + self.pos_emb[:, :T, :]  # Add positional encoding
        
        # Decode with cross-attention to condition
        decoded = self.decoder(
            tgt=action_emb,
            memory=cond_with_time
        )  # (B, T, embed_dim)
        
        # Project to action space
        noise_pred = self.output_proj(decoded)  # (B, T, action_dim)
        
        return noise_pred


class VLAModel(nn.Module):
    """
    Complete VLA model: observation encoder + diffusion transformer.
    """
    
    def __init__(
        self,
        obs_encoder: SimpleObsEncoder,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.obs_encoder = obs_encoder
        self.diffusion_transformer = DiffusionTransformer(
            action_dim=action_dim,
            horizon=horizon,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.device = device
        
        self.to(device)
    
    def forward(self, noisy_actions, timesteps, obs_dict):
        """
        Args:
            noisy_actions: (B, T, action_dim)
            timesteps: (B,) or scalar
            obs_dict: dict with image, qpos, text for first n_obs_steps
        
        Returns:
            noise_pred: (B, T, action_dim)
        """
        # Handle scalar timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=self.device).expand(noisy_actions.shape[0])
        elif len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0).expand(noisy_actions.shape[0])
        
        # Encode observations
        obs_embeddings = self.obs_encoder(obs_dict)  # (B, T_obs, embed_dim)
        
        # Predict noise
        noise_pred = self.diffusion_transformer(
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            condition=obs_embeddings
        )
        
        return noise_pred