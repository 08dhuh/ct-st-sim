from ct_core.pipeline_utils import process_params
def update_session_params():
    import streamlit as st

    theta_tuple = (
        st.session_state.theta_count,
        st.session_state.theta_max,
        st.session_state.theta_min
    )
    phi_tuple = (
        st.session_state.phi_count,
        st.session_state.phi_max,
        st.session_state.phi_min
    )
    args = dict(
        theta_tuple=theta_tuple,
        phi_tuple=phi_tuple,
        x_range=st.session_state.x_range,
        y_range=st.session_state.y_range,
        grid_count=st.session_state.grid_count,
        is_beam_angular = st.session_state.is_beam_angular,
        radius=st.session_state.radius,
        width = st.session_state.beam_width,
        width_mode = st.session_state.width_mode
        

    )

    theta, phi, mesh, width_args, strength, recon_array = process_params(**args)
    st.session_state.update({
        "theta_space": theta,
        "phi_space": phi,
        "xy_mesh": mesh,
        "common_width_kwargs": width_args,
        "beam_strength": strength,
        "recon_array": recon_array
    })