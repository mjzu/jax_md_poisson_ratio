#@title Imports and utility code
#!pip install https://github.com/cgoodri/jax-md/archive/elasticity.zip

import os
import numpy as onp

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, grad, vmap

import jax.scipy as jsp

from jax_md import space, energy, smap, util, elasticity, quantity
#from jax_md.colab_tools import renderer

f32 = jnp.float32
f64 = jnp.float64

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
#import seaborn as sns
#sns.set_style(style='white')

def format_plot(x, y):
  plt.grid(True)
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1.0, 0.7)):
  plt.gcf().set_size_inches(
     shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
     shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()
  plt.show()

def draw_2dsystem(R, box_size, marker_size, color=None):
  if color == None:
     color = [64 / 256] * 3
  ms = marker_size / box_size
  R = onp.array(R)

  marker_style = dict(
    linestyle='none',
    markeredgewidth=3,
    marker='o',
    markersize=ms,
    color=color,
    fillstyle='none')
  
  plt.plot(R[:, 0], R[:, 1], **marker_style)

  plt.xlim([0, box_size])
  plt.ylim([0, box_size])
  plt.axis('off')

def draw_3dsystem(R, box_size, marker_size, color=None):
  if color == None:
     color = [64 / 256] * 3
  ms = marker_size / box_size
  R = onp.array(R)

  fig = plt.figure(figsize=(12,12))
  ax = fig.add_subplot(projection='3d')

  ax.scatter(R[:,0], R[:,1], R[:,2], )
  plt.show()

def read_xyd(fname):
  R = []
  diameters = []
  if not os.path.isfile(fname):
    raise IOError("This file '{}' does not exist.".format(fname))
  f = open(fname, "r")
  while True:
    xyd = f.readline()
    if not xyd:
      break
    x, y, d = xyd.split()
    R.append([float(x), float(y)])
    diameters.extend([float(d)])
  return jnp.array(R, dtype=f64), jnp.array(diameters, dtype=f64)

@jit
def _vector2symmat(v, zeros):
  n = zeros.shape[0]
  assert v.shape == (
    n * (n + 1) / 2,
  ), f"The input must have shape jnp.int16(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2) = {(n * (n + 1) / 2,)}, got {v.shape} instead."
  ind = jnp.triu_indices(n)
  return zeros.at[ind].set(v).at[(ind[1], ind[0])].set(v)

@jit
def vector2dsymmat(v):
  """ Convert a vector into a symmetric matrix.
  Args:
    v: vector of length (n*(n+1)/2,)
  Return:
    symmetric matrix m of shape (n,n) that satisfies
    m[jnp.triu_indices_from(m)] == v
  Example:
    v = jnp.array([0,1,2,3,4,5])
    returns: [[ 0, 1, 2],
              1, 3, 4],
              2, 4, 5]]
  """
  n = int(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2)
  return _vector2symmat(v, jnp.zeros((n, n), dtype=v.dtype))

def diameters_to_sigma_matrix(diameters_vec):
  return vmap(vmap(lambda d1,d2: (d1 + d2) * 0.5, in_axes=(None, 0)),
              in_axes=(0, None))(diameters_vec, diameters_vec)

def B_to_epsilon_matrix(B_vec):
  return vmap(vmap(lambda B1, B2: jnp.sqrt(B1 * B2), in_axes=(None, 0)),
                       in_axes=(0, None))(B_vec, B_vec)

def stress_tensor(R, displacement, potential, box, **kwargs):
  volume = box.diagonal().prod()
  dR = space.map_product(displacement)(R, R)
  dr = space.distance(dR)

#  kwargs = smap._kwargs_to_parameters(None, **kwargs)
  dUdr = vmap(vmap(grad(potential)))(dr, **kwargs)
  temp = vmap(vmap(lambda s, m: s*m))(smap._diagonal_mask(dUdr/dr), 
              jnp.einsum('abi,abj->abij', dR, dR))
  return util.high_precision_sum(temp, axis=(0, 1)) * f32(0.5) / volume

## calculate the excess coordinate number \delt Z = Z - Z_iso ##
## where Z_iso = 2d - 2d/N is the isotropic coordinate number ##
def get_coordinate_number(displacement_or_metric, R, sigma, species=None):
  r_cutoff = 1.0
#  if (species is not None):
#    # convert jax.array into numpy.array to speed up
#    sigma_onp = onp.array(sigma) * r_cutoff
#    tmp_onp_array = [[sigma_onp[i, j] for i in species] for j in species]
#    dr_cutoff = jnp.array(tmp_onp_array)
#  else:
#    dr_cutoff = sigma * r_cutoff
  dr_cutoff = sigma * r_cutoff

  metric = space.map_product(space.canonicalize_displacement_or_metric(displacement_or_metric))
  dr = metric(R, R)

  coordinate_metric = jnp.where(dr<=dr_cutoff, 1, 0) - jnp.eye(R.shape[0], dtype=jnp.int32)
  coordinate_number = jnp.sum(coordinate_metric, axis=1).tolist()
  coordinate_list = []
  for i in range(R.shape[0]):
    coordinate_list_i = []
    for j in range(R.shape[0]):
      if coordinate_metric[i,j] > 0:
        coordinate_list_i.append(j)
    coordinate_list.append(coordinate_list_i)

  ave_coordinates = 0.0
  non_rattler = 0
  for i in range(len(coordinate_number)):
    if coordinate_number[i] >= 3:
      ave_coordinates += coordinate_number[i]
      non_rattler += 1
  if non_rattler > 0:
    ave_coordinates = ave_coordinates / float(non_rattler)
  else:
    ave_coordinates = 0.0

  return coordinate_list, ave_coordinates, non_rattler

key = random.PRNGKey(0)
