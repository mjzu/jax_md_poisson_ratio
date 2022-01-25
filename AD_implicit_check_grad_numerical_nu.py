###################################
# @name : AD_Poisson_ratio.py
# @author : mzu
# @created date : 10/01/22
# @function : checking against numerical differences for poisson ratio with initial random packing
# @ref: https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/implicit_differentiation.ipynb#scrollTo=9YB8Qr2nOL5B
###################################
import time
import sys

import jax
import jax.numpy as jnp
import jaxopt.implicit_diff
from jax.config import config

config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, vmap

from jax_md import space, energy, util, quantity, elasticity

from jaxopt.implicit_diff import custom_root, custom_fixed_point

from minimization import run_minimization_scan, run_minimization_while

from utils import diameters_to_sigma_matrix, get_coordinate_number, vector2dsymmat
from utils import draw_2dsystem, finalize_plot

f32 = jnp.float32
f64 = jnp.float64
Array = util.Array

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

key = random.PRNGKey(1)

# N: number of particles
N = 16
# dimension: dimension of the system
dimension = 2
# alpha: exponential parameter of Morse potential
alpha = 6.0

# density: number density = N / box_volume
density = 1.2

# n_species: number of species
n_species = 2
# N_s: number of particles of each specie
N_s = N // n_species
species_seed = jnp.arange(n_species)
species_vec = jnp.repeat(species_seed, N_s)

# setup species, diameters and binding energy B
diameters_seed = jnp.array([1.0, 1.4])
B_seed = jnp.array([0.01, 0.02, 0.04])

# define a set of parameters
param_dict = {"diameters_seed": diameters_seed, "B_seed": B_seed}

# initialization of box size and particle positions
box_size = quantity.box_size_at_number_density(N, density, dimension)
box = box_size * jnp.eye(dimension)
displacement, shift = space.periodic_general(box, fractional_coordinates=True)

key, split = random.split(key)
R_init = random.uniform(split, (N, dimension), minval=0.0, maxval=box_size, dtype=f64)

# define energy function
def energy_function(params):
    # setup sigmas and epsilons matrix
    sigmas_matrix = diameters_to_sigma_matrix(params["diameters_seed"])
    B_matrix = vector2dsymmat(params["B_seed"])
    return energy.morse_pair(displacement,
                                  species=species_vec, alpha=alpha,
                                  sigma=sigmas_matrix, epsilon=B_matrix)

# test the minimization steps
energy_fn = energy_function(param_dict)
force_fn = jit(quantity.force(energy_fn))
min_style = 1
R, max_grad, niters = run_minimization_while(energy_fn, R_init, shift, min_style)
print("Maximum force for initial packing: " + str(max_grad))
print("Minimization steps: " + str(niters))

num_steps = niters * 2
# define a function in order to use jax.grad
# ref@: https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/implicit_differentiation.ipynb

def explicit_diff_nu(params, R_init, num_steps):

    sigmas_matrix = diameters_to_sigma_matrix(params["diameters_seed"])
    B_matrix = vector2dsymmat(params["B_seed"])
    energy_fn_exp = energy.morse_pair(displacement,
                                  species=species_vec, alpha=alpha,
                                  sigma=sigmas_matrix, epsilon=B_matrix)

#    energy_fn_exp = energy_function(params)

    force_fn_exp = jit(quantity.force(energy_fn_exp))

    # we need to use a scan instead of a while loop in order to use jax.grad
    solver = lambda f, x: run_minimization_scan(f, x, shift, min_style=min_style, num_steps=num_steps)[0]
    R_final = solver(force_fn_exp, R_init)
#    solver = lambda f, x: run_minimization_while(f, x, shift, min_style=2)[0]
#    R_final = solver(energy_fn_exp, R_init)

    EMT_fn_exp = jit(elasticity.athermal_moduli(energy_fn_exp))
    C = EMT_fn_exp(R_final, box)
    Poisson_ratio = elasticity.extract_isotropic_moduli(C)['nu']

    return Poisson_ratio

def implicit_diff_nu(params, R_init, box):

    displacement, shift = space.periodic_general(box, fractional_coordinates=True)
    sigmas_matrix = diameters_to_sigma_matrix(params["diameters_seed"])
    B_matrix = vector2dsymmat(params["B_seed"])
    energy_fn_imp = energy.morse_pair(displacement,
                                  species=species_vec, alpha=alpha,
                                  sigma=sigmas_matrix, epsilon=B_matrix)

#    energy_fn_imp = energy_function(params)

    force_fn_imp = jit(quantity.force(energy_fn_imp))

    # wrap force_fn with a lax.stop_gradient to prevent a CustomVJPException.
    no_grad_force_fn = jit(lambda x: lax.stop_gradient(force_fn_imp(x)))

    # Make the dependence on the variables we want to differentiate explicit.
    explicit_force_fn = jit(lambda R, p: force_fn_imp(R, **p))

    def solver(params, x):
        # params are unused
        del params
        # need to use no_grad_force_fn
        return run_minimization_while(no_grad_force_fn, x, shift, min_style=min_style)[0]

    decorated_solver = custom_root(explicit_force_fn)(solver)

    R_final = decorated_solver(None, R_init)

    EMT_fn_imp = jit(elasticity.athermal_moduli(energy_fn_imp))
    C = EMT_fn_imp(R_final, box)
    nu = elasticity.extract_isotropic_moduli(C)['nu']

    return nu

#(exp_nu0, exp_R_final), exp_nu_grad = jax.value_and_grad(explicit_diff_nu, has_aux=True)(
#    param_dict, R_init, num_steps=num_steps)

#(imp_nu0, imp_R_final), imp_nu_grad = jax.value_and_grad(implicit_diff_nu, has_aux=True)(
#    param_dict, exp_R_final)

#exp_nu_grad = jax.jacfwd(explicit_diff_nu)(
#    param_dict, R_init, num_steps=num_steps)

imp_nu_grad= jax.grad(implicit_diff_nu)(
    param_dict, R_init, box)
#print(jax.tree_map(jnp.allclose, exp_nu_grad, imp_nu_grad))
print("Gradient of nu:")
print("diameters:", imp_nu_grad["diameters_seed"])
print("B:", imp_nu_grad["B_seed"])
exit()

diameters_seed0 = param_dict["diameters_seed"]
B_seed0 = param_dict["B_seed"]

# create unitvectors in a random direction
key, subkey = random.split(key)
vec_diameters = random.normal(subkey, diameters_seed0.shape)
unitvec_diameters = vec_diameters / jnp.sqrt(jnp.vdot(vec_diameters, vec_diameters))
vec_B = random.normal(subkey, B_seed0.shape)
unitvec_B = vec_B / jnp.sqrt(jnp.vdot(vec_B, vec_B))

# gradient of poisson ratio with original parameters
imp_nu_autodiff_diameter = jnp.vdot(imp_nu_grad["diameters_seed"], unitvec_diameters)
imp_nu_autodiff_B = jnp.vdot(imp_nu_grad["B_seed"], unitvec_B)
imp_nu_autodiff_diameter_B = jnp.vdot(imp_nu_grad["diameters_seed"], unitvec_diameters) + jnp.vdot(imp_nu_grad["B_seed"], unitvec_B)

exp_nu_autodiff_diameter = jnp.vdot(exp_nu_grad["diameters_seed"], unitvec_diameters)
exp_nu_autodiff_B = jnp.vdot(exp_nu_grad["B_seed"], unitvec_B)
exp_nu_autodiff_diameter_B = jnp.vdot(exp_nu_grad["diameters_seed"], unitvec_diameters) + jnp.vdot(exp_nu_grad["B_seed"], unitvec_B)

exp_numerical_nu = []
imp_numerical_nu = []
delt_l = jnp.logspace(-7, -1, num=10)

for dp in delt_l:
    B_seed1 = B_seed0 + unitvec_B * dp
    diameters_seed1 = diameters_seed0 + unitvec_diameters * dp

    param_dict1 = {"diameters_seed": diameters_seed1, "B_seed": B_seed0}
    param_dict2 = {"diameters_seed": diameters_seed0, "B_seed": B_seed1}
    param_dict3 = {"diameters_seed": diameters_seed1, "B_seed": B_seed1}

    # energy and poisson ratio with finite differences in parameters
    exp_nu1 = explicit_diff_nu(param_dict1, R_init, num_steps=num_steps)[0]
    exp_nu2 = explicit_diff_nu(param_dict2, R_init, num_steps=num_steps)[0]
    exp_nu3 = explicit_diff_nu(param_dict3, R_init, num_steps=num_steps)[0]

    imp_nu1 = implicit_diff_nu(param_dict1, R_init)[0]
    imp_nu2 = implicit_diff_nu(param_dict2, R_init)[0]
    imp_nu3 = implicit_diff_nu(param_dict3, R_init)[0]

    exp_numerical_nu += [[exp_nu1-exp_nu0, exp_nu2-exp_nu0, exp_nu3-exp_nu0]]
    imp_numerical_nu += [[imp_nu1-imp_nu0, imp_nu2-imp_nu0, imp_nu3-imp_nu0]]

resfile = "/Users/mengjiezu/PycharmProjects/jax_md_mp/src/check_autodiff_nu.dat"
print('unit vector:')
print(unitvec_diameters)
print(unitvec_B)

print("Results with implicit differentiation:")
print("nu:{:.16f}".format(imp_nu0))
print("Gradient of nu:")
print("diameters:", imp_nu_grad["diameters_seed"])
print("B:", imp_nu_grad["B_seed"])
print('gradient of nu wrt diameter:{:.16f}'.format(imp_nu_autodiff_diameter))
print('gradient of nu wrt B:{:.16f}'.format(imp_nu_autodiff_B))
print('gradient of nu wrt params:{:.16f}\n'.format(imp_nu_autodiff_diameter_B))

print("Results with explicit differentiation:")
print("nu:{:.16f}".format(exp_nu0))
print("Gradient of nu:")
print("diameters:", exp_nu_grad["diameters_seed"])
print("B:", exp_nu_grad["B_seed"])
print('gradient of nu wrt diameter:{:.16f}'.format(exp_nu_autodiff_diameter))
print('gradient of nu wrt B:{:.16f}'.format(exp_nu_autodiff_B))
print('gradient of nu wrt params:{:.16f}\n'.format(exp_nu_autodiff_diameter_B))

plotfile = "/Users/mengjiezu/PycharmProjects/jax_md_mp/src/dp_dnu_imp.dat"
f=open(plotfile, 'w')
f.write('#dp\tdnu_diameter\tdnu_B\tdnu_diameter_B\n')
f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(0.0, imp_nu0, imp_nu0, imp_nu0))
for i, dnu in enumerate(imp_numerical_nu):
    f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(delt_l[i], dnu[0], dnu[1], dnu[2]))
f.close()

plotfile = "/Users/mengjiezu/PycharmProjects/jax_md_mp/src/dp_dnu_exp.dat"
f=open(plotfile, 'w')
f.write('#dp\tdnu_diameter\tdnu_B\tdnu_diameter_B\n')
f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(0.0, exp_nu0, exp_nu0, exp_nu0))
for i, dnu in enumerate(exp_numerical_nu):
    f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(delt_l[i], dnu[0], dnu[1], dnu[2]))
f.close()

