import numpy as np
import streamlit as st
from ct_core.back_propagation import *

from ct_core.pipeline_utils import signal_readout_iter, sim_params_to_grid, phi_width, to_uint8


@st.cache_data
def run_objects_generation(params_tuple: tuple[tuple[float]],
                           gridnum: int = 250,
                           x_range: tuple[float, float] = (-5, 5),
                           y_range: tuple[float, float] = (-5, 5),
                           # plot=False,
                           white_bg: bool = True):
    """_summary_

    Args:
        params_tuple (tuple[tuple[float]]): _description_
        gridnum (int, optional): _description_. Defaults to 250.
        x_range (tuple[float, float], optional): _description_. Defaults to (-5, 5).
        y_range (tuple[float, float], optional): _description_. Defaults to (-5, 5).
        white_bg (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
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
    """one iteration of back propagation for a given beam.

    Args:
        recon_array (np.ndarray): _description_
        theta (float): _description_
        phi (float): _description_
        radius (float): _description_
        theta_count (int): _description_
        phi_count (int): _description_
        theta_range (int): _description_
        phi_range (int): _description_
        object_rows (np.ndarray): _description_
        xy_mesh (tuple[np.ndarray, np.ndarray]): _description_
        grid_count (int): _description_
        strength (float): _description_
        x_range (tuple[float, float]): _description_
        y_range (tuple[float, float]): _description_
        is_beam_angular (bool): _description_
        width_mode (str): _description_
        beam_width (float): _description_
        filter_mode (str): _description_

    Returns:
        tuple[np.ndarray, np.ndarray, bool]: _description_
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
            #recon_array = recon_array + strength * ray_array / theta_count
            np.add(recon_array, (strength * ray_array.astype(np.float32)) / theta_count, out=recon_array)

        case _:
            return recon_array, np.zeros_like(recon_array), False
    return recon_array, ray_array, True


@st.cache_data
def compute_recon_array_series(theta_space, 
                               phi_space, 
                               object_rows, 
                               xy_mesh,
                                grid_count,
                               beam_strength, 
                               x_range, 
                               y_range, 
                               width_mode, 
                               filter_mode, 
                               common_width_kwargs,
                               )-> tuple[np.ndarray, np.ndarray]:
    recon_array = np.zeros_like(xy_mesh[0], dtype=np.float32)
    recon_array_series = []
    for theta in theta_space:
        for phi in phi_space:
            temp_recon_array, ray_array, updated = run_back_propagation_iter(
                #recon_array=st.session_state.recon_array,
                recon_array=recon_array,
                theta=theta,
                phi=phi,
                object_rows=object_rows,
                xy_mesh=xy_mesh,
                grid_count=grid_count,
                strength=beam_strength,
                x_range=x_range,
                y_range=y_range,
                width_mode=width_mode,
                filter_mode=filter_mode,
                **common_width_kwargs
            )

            #st.session_state.recon_array = recon_array
            if updated:
                recon = temp_recon_array.copy().astype(np.float32) 
                beam = ray_array.astype(np.float32) 
                frame = recon + beam
                recon_array_series.append(frame)

    return recon_array, recon_array_series


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
                                    #geo: str = 'a',
                                    is_angular:bool = True,
                                    width_val: float = 0.1,
                                    array_flattening=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        params_tuple (_type_): _description_
        grid_count (int, optional): _description_. Defaults to 250.
        x_range (tuple, optional): _description_. Defaults to (-5, 5).
        y_range (tuple, optional): _description_. Defaults to (-5, 5).
        theta_tuple (tuple[int, float, float], optional): _description_. Defaults to ( 100, 2 * np.pi, 0).
        phi_tuple (tuple[int, float, float], optional): _description_. Defaults to ( 50, np.deg2rad(18), np.deg2rad(-18)).
        radius (float, optional): _description_. Defaults to 17.0.
        mode (str, optional): _description_. Defaults to 'static'.
        geo (str, optional): _description_. Defaults to 'a'.
        width_val (float, optional): _description_. Defaults to 0.1.
        array_flattening (bool, optional): _description_. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: final recon array and the generated CT scanning frames
    """
    theta_space = np.linspace(theta_tuple[1], theta_tuple[2], theta_tuple[0])
    phi_space = np.linspace(phi_tuple[1], phi_tuple[2], phi_tuple[0])

    xy_mesh = generate_mesh_grid(x_range, y_range, grid_count)
    theta_range = theta_tuple[2] - theta_tuple[1]
    phi_range = phi_tuple[2] - phi_tuple[1]

    #is_angular = geo == 'a'
    width = phi_width(mode, is_angular, width_val, 0, 0,
                      phi_tuple[0], theta_tuple[0], phi_range, theta_range, radius)

    ray_arrays = ray_id_vectorised(theta_space, phi_space, radius, grid_count,
                                  is_beam_angular=is_angular, xy_mesh=xy_mesh, width=width)

    mask = compute_signal_mask(params_tuple, theta_space, phi_space, radius)[
        :, :, None, None]
    ray_arrays_masked = ray_arrays * mask
    recon_array_series = back_propagation_from_rays(
        ray_arrays_masked, strength=1.0, theta_count=theta_tuple[0])
    if array_flattening:
        ray_arrays = ray_arrays.reshape(-1, grid_count, grid_count)
        recon_array_series = recon_array_series.reshape(-1, grid_count, grid_count)
    #return ray_arrays, recon_array_series
    return recon_array_series[-1], ray_arrays + recon_array_series
