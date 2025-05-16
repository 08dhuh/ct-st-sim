import streamlit as st
import numpy as np

from dataclasses import dataclass
from ct_core.ct_sim_main import compute_recon_array_series
from ct_core.pipeline_utils import process_params
from config import USER_INPUT_PARAM_NAMES, DEFAULT_PARAM_VALUES


@dataclass(frozen=True, slots=True)
class CTInputParams:
    theta_count: int
    theta_min: float
    theta_max: float
    phi_count: int
    phi_min: float
    phi_max: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_count: int
    is_beam_angular: bool
    radius: float
    beam_width: float
    width_mode: str
    filter_mode: str

    @classmethod
    def from_session_state(cls, st):
        return cls(**{field: st.session_state[field] for field in USER_INPUT_PARAM_NAMES})
    

    def hash_values(self):
        return hash(self)
    
    def detect_hash_change(self, st):
        current_hash = self.hash_values()
        previous_hash = st.session_state.get("ct_input_hash", None)
        if previous_hash != current_hash:
            st.session_state['ct_input_hash'] = current_hash
            return True
        return False
    
def initialise_session_state():
    """
    initialise required streamlit session state varibales
    """  
    initialise_default_session_state()
    update_derived_state()
    initialise_result_session_state()
    initialise_filter_session_state()
    if 'viewport_data' not in st.session_state:
        st.session_state.viewport_data = _generate_empty_recon_array()
    if 'viewport_data_series' not in st.session_state:
        st.session_state.viewport_data_series = []
    
def initialise_default_session_state():
    for key, value in DEFAULT_PARAM_VALUES.items():
        st.session_state.setdefault(key, value)



def initialise_filter_session_state():
    if 'filter_cutoff' not in st.session_state:
        st.session_state.filter_cutoff = 0.0
    if 'soft_cutoff' not in st.session_state:
        st.session_state.soft_cutoff = False

def initialise_result_session_state():
    st.session_state.update({
        'recon_array': _generate_empty_recon_array(),
        'recon_array_series':[]
    })

def _generate_empty_recon_array(grid_count:int | None = None) -> np.ndarray:
    """returns an empty recon array

    Raises:
        ValueError: _description_
    """
    if grid_count:
        return np.zeros((grid_count, grid_count))
    if 'xy_mesh' not in st.session_state:
        raise ValueError("xy_mesh must be initialized before recon_array.")

    return np.zeros_like(st.session_state.xy_mesh[0])
    


def update_derived_state():
    """
    use after initialise_default_session_params
    """
    x_range = (st.session_state.x_min, st.session_state.x_max)
    y_range = (st.session_state.y_min, st.session_state.y_max)

    theta_tuple = (
        st.session_state.theta_count,
        np.deg2rad(st.session_state.theta_min),
        np.deg2rad(st.session_state.theta_max)
    )
    phi_tuple = (
        st.session_state.phi_count,
        np.deg2rad(st.session_state.phi_min),
        np.deg2rad(st.session_state.phi_max)
    )
    args = dict(
        theta_tuple=theta_tuple,
        phi_tuple=phi_tuple,
        x_range=x_range,
        y_range=y_range,
        grid_count=st.session_state.grid_count,
        is_beam_angular = st.session_state.is_beam_angular,
        radius=st.session_state.radius,
        beam_width = st.session_state.beam_width,
        width_mode = st.session_state.width_mode        

    )

    theta, phi, mesh, common_width_kwargs, strength, recon_array = process_params(**args) 
    st.session_state.update({
        "x_range": x_range,
        "y_range": y_range,
        "theta_space": theta,
        "phi_space": phi,
        "xy_mesh": mesh,
        "common_width_kwargs": common_width_kwargs,
        "beam_strength": strength,

    })




def st_back_propagation():
    recon_array, recon_array_series = compute_recon_array_series(
        theta_space=st.session_state.theta_space,
        phi_space=st.session_state.phi_space,
        object_rows=st.session_state.object_rows,
        xy_mesh=st.session_state.xy_mesh,
        grid_count=st.session_state.grid_count,
        beam_strength=st.session_state.beam_strength,
        x_range=st.session_state.x_range,
        y_range=st.session_state.y_range,
        width_mode=st.session_state.width_mode,
        filter_mode=st.session_state.filter_mode,
        common_width_kwargs=st.session_state.common_width_kwargs,
    )

    st.session_state.recon_array = recon_array
    st.session_state.recon_array_series = recon_array_series


def st_radon_transform():
    #TODO: 
    pass



def cmap(white_bg:bool):
    return 'gray' if white_bg else 'gray_r'


def apply_cutoff_filter(array: np.ndarray, 
                        threshold: float, 
                        soft: bool = False) -> np.ndarray:
    if np.max(array) == 0:
        return np.zeros_like(array)
    normalised_arr = array / np.max(array)
    if soft:
        result = np.clip((normalised_arr - threshold) / (1 - threshold), 0, 1)
    else:
        result = np.where(normalised_arr >= threshold, normalised_arr, 0)
    return result
