import math
from PIL import Image, ImageDraw, ImageFilter
import random
import os
import sys
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

dir = './dataset'

def xy_rand(xy, size):
  xy_new = []
  for n in range(len(xy)):
    xy_temp = [0, 0]
    for i in range(2):
      xy_temp[i] = xy[n][i] + random.uniform(-1.0, 1.0) * size[i]
    xy_new.append((xy_temp[0], xy_temp[1]))
  return xy_new

def xy_move(xy, size):
  xy_new = []
  for n in range(len(xy)):
    xy_temp = [0, 0]
    for i in range(2):
      xy_temp[i] = xy[n][i] + size[i]
    xy_new.append((xy_temp[0], xy_temp[1]))
  return xy_new

def handwriting(draw, xy, wi):
  nlines = 10
  xi, yi = xy[0]
  xj, yj = xy[1]
  dx, dy = [(xj - xi) / nlines, (yj - yi) / nlines]
  for n in range(nlines):
    x0, y0 = [xi + dx * n, yi + dy * n]
    x1, y1 = [xi + dx * (n + 1), yi + dy * (n + 1)]
    rgb = random.randrange(64)
    draw.line(xy_rand([(x0, y0), (x1, y1)], [2, 2]), fill = (rgb, rgb, rgb), width = wi)

def draw_hexa(x, y, size, angle):
  img = Image.new('RGB', (size, size), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  xi = [x + size / 2 * (1 + math.cos(2 * math.pi * n / 6)) for n in range(0, 7)]
  yi = [y + size / 2 * (1 + math.sin(2 * math.pi * n / 6)) for n in range(0, 7)]
  xy = [(xi[n], yi[n]) for n in range(7)]
  rgb = random.randrange(196)
  draw.polygon(xy_rand(xy, [3, 3]), fill = (rgb, rgb, rgb))
  img = img.rotate(angle, fillcolor=(255, 255, 255), expand = False)
  return img

def draw_line(order, x, y, size, angle):
  img = Image.new('RGB', (size, size), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  wi = int(size * 0.1)
  xi = [x + size * 0.1, x + size * 0.9]
  yi = [y + size / 2.0, y + size / 2.0]
  xy = [(xi[n], yi[n]) for n in range(2)]
  if order == 1:
    handwriting(draw, xy_rand(xy, [2, 2]), int(wi + random.uniform(-2.0, 2.0)))
  elif order == 2:
    handwriting(draw, xy_rand(xy_move(xy, [0, +wi * 2 + random.uniform(-2.0, 2.0)]), [8, 2]), int(wi + random.uniform(-2.0, 2.0)))
    handwriting(draw, xy_rand(xy_move(xy, [0, -wi * 2 + random.uniform(-2.0, 2.0)]), [8, 2]), int(wi + random.uniform(-2.0, 2.0)))
  elif order == 3:
    handwriting(draw, xy_rand(xy_move(xy, [0, +wi * 3.5 + random.uniform(-2.0, 2.0)]), [8, 2]), int(wi + random.uniform(-2.0, 2.0)))
    handwriting(draw, xy_rand(xy_move(xy, [0,       0]), [8, 2]), int(wi + random.uniform(-1, 1)))
    handwriting(draw, xy_rand(xy_move(xy, [0, -wi * 3.5 + random.uniform(-2.0, 2.0)]), [8, 2]), int(wi + random.uniform(-2.0, 2.0)))
  img = img.rotate(angle, fillcolor=(255, 255, 255), expand = False)
  return img

def add_noise(draw, row, col):
  # white
  xy = tuple([(np.random.randint(0, col), np.random.randint(0, row)) for n in range(int(row * col * random.uniform(0.01, 0.05)))])
  draw.point(xy, fill = (255, 255, 255))
  # black
  xy = tuple([(np.random.randint(0, col), np.random.randint(0, row)) for n in range(int(row * col * random.uniform(0.01, 0.05)))])
  draw.point(xy, fill = (0, 0, 0))
  return img

def train_valid_test_split(*arrays, valid_size: float, test_size: float, **kwargs):
  first_split = train_test_split(*arrays, test_size=test_size, **kwargs)
  testing_data = first_split[1::2]
  if valid_size == 0:
    training_data = first_split[::2]
    validation_data = []
  else:
    training_validation_data = train_test_split(*first_split[::2], test_size=(valid_size / (1 - test_size)), **kwargs)
    training_data = training_validation_data[::2]
    validation_data = training_validation_data[1::2]
  return training_data + validation_data + testing_data

def makedirs(dir):
  for n in ['train', 'valid', 'test']:
    for i in ['images', 'labels']:
      os.makedirs(f'{dir}/{n}/{i}', exist_ok = True)
      shutil.rmtree(f'{dir}/{n}/{i}')
      os.makedirs(f'{dir}/{n}/{i}')
  with open(f'{dir}/data.yaml', 'w') as fw:
    fw.write('train: ../dataset/train/images\n')
    fw.write('val: ../dataset/valid/images\n')
    fw.write('test: ../dataset/test/images\n\n')
    fw.write('nc : 4\n')
    fw.write("names: ['atom', 'single', 'double', 'triple']\n")

makedirs(dir)

nimages = 10
if len(sys.argv) > 1:
  nimages = int(sys.argv[1])
else:
  sys.exit()

images = []
nitems = 6000
for n in range(nitems):
  w = random.uniform(1.0, 1.5)
  images.append([draw_hexa(   0, 0, int(64 * w), 360 * (-1.0 + 1.0 / nitems * n)), 0, w])
  w = random.uniform(1.0, 1.5)
  images.append([draw_line(1, 0, 0, int(64 * w), 360 * (-1.0 + 1.0 / nitems * n)), 1, w])
  w = random.uniform(1.0, 1.5)
  images.append([draw_line(2, 0, 0, int(64 * w), 360 * (-1.0 + 1.0 / nitems * n)), 2, w])
  w = random.uniform(1.0, 1.5)
  images.append([draw_line(3, 0, 0, int(64 * w), 360 * (-1.0 + 1.0 / nitems * n)), 3, w])
random.shuffle(images)

imgnum = {'train' : [], 'valid' : [], 'test' : []}
imgnum['train'], imgnum['valid'], imgnum['test'] = train_valid_test_split(
  range(nimages), valid_size = 0.2, test_size = 0.1)

for type in ['train', 'valid', 'test']:
  for n in imgnum[type]:
    size = 640
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    with open('{:s}/{:s}/labels/{:03d}.txt'.format(dir, type, n), 'w') as fw:
      ddi = [random.uniform(-0.05, 0.05) for k in range(5)]
      for i in range(5):
        ddj = [random.uniform(-0.05, 0.05) for k in range(5)]
        for j in range(5):          
          num = random.randrange(len(images))
          w = images[num][2]
          sgm = (0.2 - 0.1 * w) / 2
          dx, dy = [random.uniform(-sgm, sgm), random.uniform(-sgm, sgm)]
          if n % 4 == 0:
            x, y =  [(sgm + i * 0.2 + dx) * 640, (sgm + j * 0.2 + dy) * 640]
            xi, yi = [0.1 + i * 0.2 + dx        , 0.1 + j * 0.2 + dy]
            img.paste(images[num][0], (int(x), int(y)))
            fw.write('{:1d}{:9.6f}{:9.6f}{:9.6f}{:9.6f}\n'.format(images[num][1], xi, yi, 0.1 * w, 0.1 * w))
          elif n % 4 == 1:
            if (i < 4):
              x, y =  [(sgm + i * 0.2 + dx + 0.1 + ddi[j]) * 640, (sgm + j * 0.2 + dy) * 640]
              xi, yi = [0.1 + i * 0.2 + dx + 0.1 + ddi[j]       ,  0.1 + j * 0.2 + dy]
              img.paste(images[num][0], (int(x), int(y)))
              fw.write('{:1d}{:9.6f}{:9.6f}{:9.6f}{:9.6f}\n'.format(images[num][1], xi, yi, 0.1 * w, 0.1 * w))
          elif n % 4 == 2:
            if (j < 4):
              x, y =  [(sgm + i * 0.2 + dx) * 640, (sgm + j * 0.2 + dy + 0.1 + ddj[i]) * 640]
              xi, yi = [0.1 + i * 0.2 + dx       ,  0.1 + j * 0.2 + dy + 0.1 + ddj[i]]
              img.paste(images[num][0], (int(x), int(y)))
              fw.write('{:1d}{:9.6f}{:9.6f}{:9.6f}{:9.6f}\n'.format(images[num][1], xi, yi, 0.1 * w, 0.1 * w))
          elif n % 4 == 3:
            if (i < 4) and (j < 4):
              x, y =  [(sgm + i * 0.2 + dx + 0.1 + ddi[j]) * 640, (sgm + j * 0.2 + dy + 0.1 + ddj[i]) * 640]
              xi, yi = [0.1 + i * 0.2 + dx + 0.1 + ddi[j]       ,  0.1 + j * 0.2 + dy + 0.1 + ddj[i]]
              img.paste(images[num][0], (int(x), int(y)))
              fw.write('{:1d}{:9.6f}{:9.6f}{:9.6f}{:9.6f}\n'.format(images[num][1], xi, yi, 0.1 * w, 0.1 * w))
      add_noise(draw, size, size)
      img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 2.0)))
      img.save('{:s}/{:s}/images/{:03d}.jpg'.format(dir, type, n))
