import numpy as np
import streamlit as st
from ct_core.back_propagation import *

from ct_core.pipeline_utils import signal_readout_iter, sim_params_to_grid, phi_width


@st.cache_data
def run_objects_generation(params_tuple: tuple[tuple[float]],
                           gridnum: int = 250,
                           x_range: tuple[float, float] = (-5, 5),
                           y_range: tuple[float, float] = (-5, 5),
                           # plot=False,
                           white_bg: bool = True):
    return sim_params_to_grid(params_tuple=params_tuple,
                              gridnum=gridnum,
                              x_range=x_range,
                              y_range=y_range,
                              white_bg=white_bg)


def run_back_propagation_iter(recon_array: np.ndarray,
                              theta: float,
                              phi: float,
                              radius: float,
                              theta_count: int,
                              phi_count: int,
                              # theta_count: int,
                              theta_range: int,
                              phi_range: int,

                              object_rows: np.ndarray,
                              xy_mesh: tuple[np.ndarray, np.ndarray],
                              grid_count: int,
                              strength: float,

                              x_range: tuple[float, float],
                              y_range: tuple[float, float],
                              is_beam_angular: bool,
                              width_mode: str,
                              beam_width: float,
                              filter_mode: str,
                              ) -> tuple[np.ndarray, np.ndarray, bool]:
    # theta_space,phi_space, grid_count,xy_mesh, common_width_kwargs, strength, recon_array
    """
    one iteration of back propagation for a given beam.
    """
    # ray_array = np.zeros_like(recon_array)

    if not signal_readout_iter(obj_params=object_rows,
                               theta=theta,
                               phi=phi,
                               radius=radius,
                               ):
        return recon_array, np.zeros_like(recon_array), False
    match filter_mode:
        case 'default':
            ray_array = ray_id_scalar(theta,
                                      phi,
                                      radius,
                                      grid_count,
                                      is_beam_angular,
                                      xy_mesh,
                                      x_range,
                                      y_range,
                                      phi_width(
                                          phi=phi,
                                          phi_count=phi_count,
                                          theta_count=theta_count,
                                          phi_range=phi_range,
                                          theta_range=theta_range,
                                          radius=radius,
                                          beam_width=beam_width,
                                          width_mode=width_mode,
                                          is_beam_angular=is_beam_angular
                                      )
                                      )
            recon_array = recon_array + strength * ray_array / theta_count

        case _:
            return recon_array, np.zeros_like(recon_array), False
    return recon_array, ray_array, True


@st.cache_data
def run_back_propagation_vectorised(params_tuple,
                                    grid_count: int = 250,
                                    x_range: tuple = (-5, 5),
                                    y_range: tuple = (-5, 5),
                                    theta_tuple: tuple[int, float, float] = (
                                        100, 2 * np.pi, 0),
                                    phi_tuple: tuple[int, float, float] = (
                                        50, np.deg2rad(18), np.deg2rad(-18)),
                                    radius: float = 17.0,
                                    mode: str = 'static',
                                    geo: str = 'a',
                                    width_val: float = 0.1,
                                    array_flattening=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_space = np.linspace(theta_tuple[1], theta_tuple[2], theta_tuple[0])
    phi_space = np.linspace(phi_tuple[1], phi_tuple[2], phi_tuple[0])

    xy_mesh = generate_mesh_grid(x_range, y_range, grid_count)
    theta_range = theta_tuple[2] - theta_tuple[1]
    phi_range = phi_tuple[2] - phi_tuple[1]

    is_angular = geo == 'a'
    width = phi_width(mode, is_angular, width_val, 0, 0,
                      phi_tuple[0], theta_tuple[0], phi_range, theta_range, radius)

    ray_array = ray_id_vectorised(theta_space, phi_space, radius, grid_count,
                                  is_beam_angular=is_angular, xy_mesh=xy_mesh, width=width)

    mask = compute_signal_mask(params_tuple, theta_space, phi_space, radius)[
        :, :, None, None]
    ray_array_masked = ray_array * mask
    recon_array = back_propagation_from_rays(
        ray_array_masked, strength=1.0, theta_count=theta_tuple[0])
    if array_flattening:
        ray_array = ray_array.reshape(-1, grid_count, grid_count)
        recon_array = recon_array.reshape(-1, grid_count, grid_count)
    return ray_array, recon_array
