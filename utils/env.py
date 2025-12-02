#!/usr/bin/env python3
"""
Simplified Franka Environment for Pick-and-Place Tasks

A minimal MuJoCo environment for a Franka Panda robot arm that:
- Supports pick-and-place tasks with colored blocks
- Provides RGB camera observations
- Uses simple position control
- Returns success metrics
"""

import numpy as np
import mujoco
import mujoco.viewer as viewer
from pathlib import Path


class FrankaEnv:
    """
    Minimal Franka robot environment with:
    - RGB camera observations (240x320)
    - Joint position observations (7 DOF arm)
    - Simple position control (8D: 7 joints + gripper)
    - Pick-and-place task with red and blue blocks
    """
    
    def __init__(self, xml_path=None, max_steps=200, visualize=False):
        # Locate XML file (from original codebase)
        if xml_path is None:
            script_dir = Path(__file__).parent.parent
            xml_path = script_dir / "assets" / "franka_scene.xml"
            if not xml_path.exists():
                raise FileNotFoundError(f"MuJoCo XML not found at {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.visualize = visualize
        
        # Cache body IDs
        self.red_cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_pick"
        )
        self.blue_cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_A"
        )
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        
        # Home position for robot
        self.home_q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        
        # Renderer for images
        self.renderer = mujoco.Renderer(self.model, width=320, height=240)
        
        # Viewer (optional)
        self.mj_viewer = None
        
        self.step_count = 0
        self.current_task = "pick_place"
        self.current_target_text = "red block"
    
    def reset(self, task="pick_place", target="red"):
        """Reset environment and return initial observation."""
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Set home position
        self.data.qpos[:7] = self.home_q
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = self.home_q
        self.data.ctrl[7] = 255.0  # Open gripper
        
        # Randomize cube positions
        self._randomize_cubes()
        
        # Let physics settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        # Reset counters
        self.step_count = 0
        self.current_task = task
        self.current_target_text = "red block" if target == "red" else "blue block"
        
        return self._get_obs()
    
    def _randomize_cubes(self):
        """Randomize cube positions on table."""
        # Base positions
        red_base = np.array([0.7, -0.1, 0.45])
        blue_base = np.array([0.7, 0.1, 0.45])
        
        # Add random offset
        red_pos = red_base + np.random.uniform(-0.1, 0.1, size=3)
        red_pos[2] = 0.45  # Keep z constant
        
        blue_pos = blue_base + np.random.uniform(-0.1, 0.1, size=3)
        blue_pos[2] = 0.45
        
        # Set positions
        self._set_cube_pos(self.red_cube_id, red_pos)
        self._set_cube_pos(self.blue_cube_id, blue_pos)
    
    def _set_cube_pos(self, body_id, xyz):
        """Set cube position in qpos."""
        jnt_adr = int(self.model.body_jntadr[body_id])
        qpos_adr = int(self.model.jnt_qposadr[jnt_adr])
        self.data.qpos[qpos_adr:qpos_adr + 3] = xyz
    
    def step(self, action):
        """
        Execute action and return next observation.
        
        Args:
            action: (8,) numpy array [7 arm joints + 1 gripper]
        
        Returns:
            obs: dict with image, qpos, text
            reward: float (1.0 if success, 0.0 otherwise)
            done: bool
            info: dict with success flag
        """
        # Apply action (position control)
        ctrl = np.zeros(8)
        ctrl[:7] = action[:7]  # Arm joints
        ctrl[7] = 255.0  # Gripper (always open for now)
        
        self.data.ctrl[:] = ctrl
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        
        # Update viewer if visualizing
        if self.visualize and self.mj_viewer is not None:
            self.mj_viewer.sync()
        
        # Check success
        success = self._check_success()
        done = (self.step_count >= self.max_steps) or success
        
        reward = 1.0 if success else 0.0
        
        obs = self._get_obs()
        info = {'success': success}
        
        return obs, reward, done, info
    
    def _check_success(self):
        """Check if end-effector is above target cube."""
        # Get target cube position
        if "red" in self.current_target_text:
            cube_pos = self.data.xpos[self.red_cube_id]
        else:
            cube_pos = self.data.xpos[self.blue_cube_id]
        
        # Get end-effector position
        ee_pos = self.data.xpos[self.ee_body_id]
        
        # Success: end-effector within 0.1m of position 0.2m above cube
        target_pos = cube_pos + np.array([0, 0, 0.2])
        dist = np.linalg.norm(ee_pos - target_pos)
        
        return dist < 0.1
    
    def _get_obs(self):
        """Get current observation."""
        # Render image
        self.renderer.update_scene(self.data, camera="overhead")
        image = self.renderer.render()  # (H, W, 3) uint8
        
        # Get qpos (arm joints only)
        qpos = self.data.qpos[:7].copy()  # (7,)
        
        return {
            'image': image,
            'qpos': qpos,
            'text': self.current_target_text
        }
    
    def close(self):
        """Cleanup resources."""
        if self.mj_viewer is not None:
            self.mj_viewer.close()