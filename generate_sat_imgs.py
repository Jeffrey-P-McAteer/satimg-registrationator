# /// script
# requires-python = "==3.10"
# dependencies = [
#   "blendify[utils]"
# ]
# ///

import os
import sys


def render_one_shot(out_dir, file_prefix, img_w_px, img_h_px, height_m, pitch_deg, roll_deg, sun_angle_deg):
  from blendify import scene
  from blendify.materials import PrincipledBSDFMaterial
  from blendify.colors import UniformColors
  # Add light; https://virtualhumans.mpi-inf.mpg.de/blendify/api/blendify.lights.html#blendify.lights.collection.LightsCollection.add_sun
  scene.lights.add_sun(strength=100)
  # Add camera
  scene.set_perspective_camera((img_w_px, img_h_px), fov_x=0.7, rotation=(0.82, 0.42, 0.18, 0.34), translation=(5, -5, 5))
  # Create material
  material = PrincipledBSDFMaterial()
  # Create color
  color = UniformColors((0.0, 1.0, 0.0))
  # Add cube mesh
  scene.renderables.add_cube_mesh(1.0, material, color)

  # "Ground"
  ground_color = UniformColors((1.0, 1.0, 1.0))
  scene.renderables.add_cube_mesh(10.0, material, ground_color, translation=(0, 0, -11.0))

  # Render scene
  scene.render(filepath=os.path.join(out_dir, f'{file_prefix}.png'))

REPO_ROOT = os.path.dirname(__file__)
imgs_folder = os.path.join(REPO_ROOT, 'build', 'imgs')
os.makedirs(imgs_folder, exist_ok=True)

for i in range(0, 2):
  render_one_shot(imgs_folder, f'shot_{i}', 512, 512, 0, 0, 0, 0)


