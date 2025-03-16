# /// script
# requires-python = "==3.10"
# dependencies = [
#   "opencv-python",
#   "numpy",
#   "matplotlib",
# ]
# ///


import os
import sys
import random
import glob
import time

import cv2
import numpy
import matplotlib


REPO_ROOT = os.path.dirname(__file__)
imgs_folder = os.path.join(REPO_ROOT, 'build', 'imgs')
os.makedirs(imgs_folder, exist_ok=True)

normalized_imgs = os.path.join(REPO_ROOT, 'build', 'normalized-imgs')
os.makedirs(normalized_imgs, exist_ok=True)

# Select a random image & use as base
base_img_path = random.choice([x for x in glob.glob(os.path.join(imgs_folder, '*.png'))])

print(f'base_img_path = {base_img_path}')

base_img = cv2.imread(base_img_path, cv2.IMREAD_COLOR)
base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
base_img_h, base_img_w = base_img_gray.shape

for other_img_path in glob.glob(os.path.join(imgs_folder, '*.png')):
  begin_s = time.time()
  normalized_img_path = os.path.join(normalized_imgs, os.path.basename(other_img_path))
  other_img = cv2.imread(other_img_path, cv2.IMREAD_COLOR)
  other_img_gray = cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY)

  orb_detector = cv2.ORB_create(5000)
  kp1, d1 = orb_detector.detectAndCompute(other_img_gray, None)
  kp2, d2 = orb_detector.detectAndCompute(base_img_gray, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(d1,d2)
  matches = sorted(matches, key = lambda x:x.distance)
  matches.sort(key = lambda x: x.distance)

  # Take the top 90 % matches forward
  matches = matches[:int(len(matches)*90)]
  # the number of good matches
  no_of_matches = len(matches)

  p1 = numpy.zeros((no_of_matches, 2))
  p2 = numpy.zeros((no_of_matches, 2))

  # populate the good matches
  for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

  homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
  transformed_img = cv2.warpPerspective(other_img, homography, (base_img_w, base_img_h))
  cv2.imwrite(normalized_img_path, transformed_img)

  duration_s = time.time() - begin_s
  print(f'{other_img_path} normalized in {duration_s:.2f}s')

print(f'Done!')



