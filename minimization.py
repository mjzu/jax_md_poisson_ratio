#@title running energy minimization
###################################
# @name : minimization.py
# @author : mzu
# @created date : 07/01/22
# @function : subfile including modified FIRE and program of minimization
# @ref: https://git.ist.ac.at/goodrichgroup/common_utils/-/blob/team/common_utils/minimization.py
###################################
"""
Defines a modified FIRE minimization.

The code implements the "optimization of the fast intertial relaxation engine" from [1].

[1] Julien Guénolé, Wolfram G.Nöhring, Aviral Vaid, Frédéric Houllé, Zhuocheng Xie, Aruna Prakash, and Erik Bitzek.
"Assessment and optimization of the fast inertial relaxation engine (fire) for energy minimization in atomistic
simulations and its implementation in lammps". Computational Materials Science Volume 175, 109584 (2020).
"""

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import jit, lax

from jax_md import minimize, util, quantity
from jax_md import dataclasses

f32 = jnp.float32
f64 = jnp.float64
Array = util.Array

def run_minimization_while(force_fn, R_init, shift, min_style,
                           max_grad_thresh=1e-12, max_num_steps=10000000,
                           **kwargs):
  dt_start = 0.001
  dt_max = 0.04
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(force_fn), shift, dt_start=dt_start, dt_max=dt_max, **kwargs)
  else:
      init, apply = modif_fire(jit(force_fn), shift, dt_start=dt_start, dt_max=dt_max, **kwargs)

  apply = jit(apply)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(val):
     state, i = val
     return jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def body_fn(val):
     state, i = val
     return apply(state), i + 1

  state = init(R_init)
  state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))

  return state.position, get_maxgrad(state), num_iterations

# This version is fully differentiable and internally jitted
def run_minimization_scan(force_fn, R_init, shift, min_style, num_steps=5000, **kwargs):
    dt_start = 0.001
    dt_max = 0.04
    if min_style == 1:
        init, apply = minimize.fire_descent(jit(force_fn), shift, dt_start=dt_start, dt_max=dt_max, **kwargs)
    else:
        init, apply = modif_fire(jit(force_fn), shift, dt_start=dt_start, dt_max=dt_max, **kwargs)

    apply = jit(apply)

    @jit
    def scan_fn(state, i):
        return apply(state), 0.0

    state = init(R_init)
    state, _ = lax.scan(scan_fn, state, jnp.arange(num_steps))

    return state.position, jnp.amax(jnp.abs(force_fn(state.position)))

@dataclasses.dataclass
class ModifFireState:
    position: Array
    velocity: Array
    force: Array
    dt: float
    alpha: float
    n_pos: int


def modif_fire(energy_fn, shift_fn,
               dt_start: float = 0.1,
               dt_max: float = 0.4,
               n_min: float = 5,
               f_inc: float = 1.1,
               f_dec: float = 0.5,
               alpha_start: float = 0.1,
               f_alpha: float = 0.99) -> ModifFireState:
    force = quantity.canonicalize_force(energy_fn)

    def init_fn(R: Array, **kwargs) -> ModifFireState:
        V = jnp.zeros_like(R)
        n_pos = jnp.zeros((), jnp.int32)
        F = force(R, **kwargs)
        return ModifFireState(R, V, F, dt_start, alpha_start, n_pos)

    def apply_fn(state: ModifFireState, **kwargs) -> ModifFireState:
        R, V, F_old, dt, alpha, n_pos = dataclasses.astuple(state)

        P = jnp.array(jnp.dot(jnp.reshape(F_old, (-1)), jnp.reshape(V, (-1))))

        n_pos = jnp.where(P > 0, n_pos + 1, 0)
        dt_choice = jnp.array([dt * f_inc, dt_max])
        dt = jnp.where(P > 0,
                       jnp.where(n_pos > n_min,
                                 jnp.min(dt_choice),
                                 dt),
                       dt)
        dt = jnp.where(P < 0, dt * f_dec, dt)
        alpha = jnp.where(P > 0,
                          jnp.where(n_pos > n_min,
                                    alpha * f_alpha,
                                    alpha),
                          alpha)
        alpha = jnp.where(P < 0, alpha_start, alpha)

        R = jnp.where(P < 0, shift_fn(R, -dt * V * 0.5, **kwargs), R)
        V = jnp.where(P < 0, 0, V)

        V = V + dt * F_old
        F_norm = jnp.sqrt(jnp.sum(F_old ** f32(2)) + f32(1e-6))
        V_norm = jnp.sqrt(jnp.sum(V ** f32(2)))

        V = (1.0 - alpha) * V + alpha * F_old * V_norm / F_norm
        R = shift_fn(R, dt * V, **kwargs)

        F = force(R, **kwargs)

        return ModifFireState(R, V, F, dt, alpha, n_pos)

    return init_fn, apply_fn
