# /// script
# requires-python = "==3.10"
# dependencies = [
#   "blendify[utils]",
#   "imageio",
# ]
# ///

import os
import sys

# Disable numpy warnings
import warnings
warnings.filterwarnings('ignore')

import imageio

import blendify
from blendify import scene
from blendify.materials import PrincipledBSDFMaterial, PlasticMaterial
from blendify.colors import UniformColors


def render_one_shot(out_dir, file_prefix, img_w_px, img_h_px, camera_location_xyz, sun_angle_deg):

  scene.clear()

  # Add light; https://virtualhumans.mpi-inf.mpg.de/blendify/api/blendify.lights.html#blendify.lights.collection.LightsCollection.add_sun
  scene.lights.set_background_light(strength=0.1, color=(1.0, 1.0, 1.0))
  scene.lights.add_sun(strength=3.0, rotation_mode='look_at', rotation=(-2, -2, 0), angular_diameter=0.0093)
  #scene.lights.add_point(strength=14000, shadow_soft_size=0.5, rotation_mode='look_at', rotation=(0, 0, 0), translation=(-2, -2, 28))
  scene.lights.add_point(strength=14000, shadow_soft_size=0.5, rotation_mode='look_at', rotation=(0, 0, 0), translation=(-2, -2, 28))

  # Add camera
  scene.set_perspective_camera((img_w_px, img_h_px), focal_dist=1250, rotation_mode='look_at', rotation=(0, 0, 0), translation=camera_location_xyz)

  # Create material
  material = PrincipledBSDFMaterial(
    specular=0.0, roughness=2.6, sheen_tint=0.0, ior=5.0
  )
  # material = PlasticMaterial(
  #   specular=0.1, roughness=2.6, clearcoat_roughness=0.6
  # )

  # Create color
  #color = UniformColors((0.0, 1.0, 0.0))

  # Add cube mesh(s)
  scene.renderables.add_cube_mesh(1.0, material, UniformColors((0.0, 1.0, 0.0)), translation=(0, 0, 0.48))
  scene.renderables.add_cube_mesh(0.75, material, UniformColors((0.0, 0.0, 1.0)), translation=(0, 1.75, 0.74/2.0 ))
  scene.renderables.add_cube_mesh(0.75, material, UniformColors((0.0, 0.0, 1.0)), translation=(1.75, 0.5, 0.74/2.0 ))

  # "Ground"
  ground_color = UniformColors((207/255.0, 181/255.0, 144/255.0))
  scene.renderables.add_cube_mesh(100.0, material, ground_color, translation=(0, 0, -50.0))

  # Render scene
  out_png = os.path.join(out_dir, f'{file_prefix}.png')
  #scene.render(filepath=os.path.join(out_dir, f'{file_prefix}.png'), samples=60, verbose=False, aa_filter_width=0)
  scene.preview(out_png)


REPO_ROOT = os.path.dirname(__file__)
imgs_folder = os.path.join(REPO_ROOT, 'build', 'imgs')
os.makedirs(imgs_folder, exist_ok=True)

imgs_list = []
for y in range(-12, 13):
  render_one_shot(imgs_folder, f'shot_{y}', 512, 512, (12, y/2.0, 12), 0)
  shot_img_path = os.path.join(imgs_folder, f'shot_{y}.png')
  if os.path.exists(shot_img_path):
    imgs_list.append( imageio.imread(shot_img_path) )

imageio.mimsave(os.path.join(imgs_folder, 'all.gif'), imgs_list, fps=4)

