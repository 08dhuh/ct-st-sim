import streamlit as st
import numpy as np

from dataclasses import dataclass
from ct_core.ct_sim_main import compute_recon_array_series
from ct_core.pipeline_utils import process_params
from config import USER_INPUT_PARAM_NAMES, DEFAULT_PARAM_VALUES, BP_PARAM_NAMES


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
    initialise_playback_session_state()
    init_recon_array()
    
def initialise_default_session_state():
    for key, value in DEFAULT_PARAM_VALUES.items():
        st.session_state.setdefault(key, value)

def initialise_playback_session_state():    
    st.session_state.update({
        'recon_array_series': [],
        'isPlaying': False,
        'play_index': 0
    })

def init_recon_array(flush:bool=False):
    """_summary_

    Raises:
        ValueError: _description_
    """
    if 'xy_mesh' not in st.session_state:
        raise ValueError("xy_mesh must be initialized before recon_array.")

    if 'recon_array' not in st.session_state or flush:
        st.session_state.recon_array = np.zeros_like(st.session_state.xy_mesh[0])
    


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

    theta, phi, mesh, common_width_kwargs, strength, recon_array = process_params(**args) #do not update recon_array after every st.rerun
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

    init_recon_array(flush=True)

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
    # st.session_state.update({'recon_array': recon_array,
    #                          'recon_array_series': recon_array_series})

def st_radon_transform():
    #TODO: initalisation
    pass

def st_playback():
    pass
    #TODO: write code here


#ui helper
def cmap(white_bg:bool):
    return 'Greys' if white_bg else 'Greys_r'