from ct_core.pipeline_utils import process_params
from dataclasses import dataclass
from config import USER_INPUT_PARAM_NAMES, DEFAULT_PARAM_VALUES



@dataclass(frozen=True, slots=True)
class CTInputParams:
    theta_count: int
    theta_max: float
    theta_min: float
    phi_count: int
    phi_max: float
    phi_min: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_count: int
    is_beam_angular: bool
    radius: float
    beam_width: float
    width_mode: str

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
    
def initialise_session_states():
    initialise_default_session_params()
    update_derived_params()
    
def initialise_default_session_params():
    import streamlit as st
    for key, value in DEFAULT_PARAM_VALUES.items():
        st.session_state.setdefault(key, value)


def update_derived_params():
    """
    use after initialise_default_session_params
    """
    import streamlit as st
    x_range = (st.session_state.x_min, st.session_state.x_max)
    y_range = (st.session_state.y_min, st.session_state.y_max)

    theta_tuple = (
        st.session_state.theta_count,
        st.session_state.theta_min,
        st.session_state.theta_max
    )
    phi_tuple = (
        st.session_state.phi_count,
        st.session_state.phi_min,
        st.session_state.phi_max
    )
    args = dict(
        theta_tuple=theta_tuple,
        phi_tuple=phi_tuple,
        x_range=x_range,
        y_range=y_range,
        grid_count=st.session_state.grid_count,
        is_beam_angular = st.session_state.is_beam_angular,
        radius=st.session_state.radius,
        width = st.session_state.beam_width,
        width_mode = st.session_state.width_mode        

    )

    theta, phi, mesh, width_args, strength, recon_array = process_params(**args)
    st.session_state.update({
        "x_range": x_range,
        "y_range": y_range,
        "theta_space": theta,
        "phi_space": phi,
        "xy_mesh": mesh,
        "common_width_kwargs": width_args,
        "beam_strength": strength,
        "recon_array": recon_array
    })