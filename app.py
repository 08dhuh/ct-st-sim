import streamlit as st

import matplotlib.pyplot as plt

from ct_core import ct_sim_main as ct
from config import WIDTH_MODE_MAPPING, FILTER_MODE_MAPPING, RECON_METHOD_MAPPING
from session_helpers import CTInputParams, st_back_propagation, initialise_session_state, update_derived_state, initialise_playback_session_state,cmap


# session state initialisation
initialise_session_state()


st.title('CT Scanner Simulator')

st.markdown("""
 * Use the menu at left to set plot parameters
""")

with st.sidebar:
    st.markdown("# Set Parameters")

    with st.expander("Object Geometry", expanded=True):

        for i, row in enumerate(st.session_state.object_rows):
            cols = st.columns([1, 1, 1, 0.4])

            with cols[0]:
                row[0] = st.number_input(f"x{i+1}", value=row[0], key=f"x{i}")
            with cols[1]:
                row[1] = st.number_input(f"y{i+1}", value=row[1], key=f"y{i}")
            with cols[2]:
                row[2] = st.number_input(
                    f"r{i+1}", value=row[2], min_value=0., key=f"r{i}")
            with cols[3]:
                st.write("")
                if st.button("×", key=f"delete_{i}", disabled=len(st.session_state.object_rows) <= 1):
                    st.session_state.object_rows.pop(i)
                    st.rerun()
        if st.button("➕ Add Object", disabled=len(st.session_state.object_rows) >= 4, key="add_row"):
            st.session_state.object_rows.append([0., 0., 0.])
            st.rerun()
        st.session_state.white_bg = st.checkbox('White Background',value=True)  
        obj_tuple = tuple(tuple(obj) for obj in st.session_state.object_rows)
        obj_masks = ct.run_objects_generation(params_tuple=obj_tuple,
                                              gridnum=st.session_state.grid_count,
                                              x_range=st.session_state.x_range,
                                              y_range=st.session_state.y_range,
                                              white_bg=True,
                                              )

        st.markdown('##### Preview')
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(obj_masks, cmap=cmap(st.session_state.white_bg), origin='lower', vmin=0, vmax=1)
        ax.axis('off')
        st.pyplot(fig)


    with st.expander("Image Grid Parameters", expanded=True):
        st.markdown("### X Range")
        col_x1, col_x2 = st.columns(2)


        with col_x1:
            st.session_state.x_min = st.number_input("X min",
                                                     value=st.session_state.x_min,
                                                     max_value=st.session_state.x_max - 1,
                                                     key='x_min_input',
                                                     disabled=True)
        with col_x2:
            st.session_state.x_max = st.number_input("X max",
                                                     value=st.session_state.x_max,
                                                     min_value=st.session_state.x_min + 1,
                                                     key='x_max_input',
                                                     disabled=True)
        st.markdown("### Y Range")
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            st.session_state.y_min = st.number_input("Y min",
                                                     value=st.session_state.y_min,
                                                     max_value=st.session_state.y_max - 1,
                                                     key='y_min_input',
                                                     disabled=True)
        with col_y2:
            st.session_state.y_max = st.number_input("Y max",
                                                     value=st.session_state.y_max,
                                                     min_value=st.session_state.y_min + 1,
                                                     key='y_max_input',
                                                     disabled=True)
        st.session_state.grid_count = st.number_input(
            label="grid count",
            min_value=100,
            max_value=700,
            value=int(st.session_state.grid_count),
            step=50)

    with st.expander("Reconstruction Parameters", expanded=True):
        st.markdown("### θ")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.session_state.theta_count = st.number_input(
                "θ count", value=st.session_state.theta_count, min_value=1)

        with col_t2:
            st.session_state.theta_min = st.number_input(
                "θ min",
                value=st.session_state.theta_min,
                min_value=0.,
                max_value=st.session_state.theta_max,
                key="theta_min_input"
            )

        with col_t3:
            st.session_state.theta_max = st.number_input(
                "θ max",
                value=st.session_state.theta_max,
                min_value=st.session_state.theta_min,
                max_value=360.,
                key="theta_max_input"
            )

        st.markdown("### φ")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.session_state.phi_count = st.number_input(
                "φ count", value=st.session_state.phi_count, min_value=1)
        with col_p2:
            st.session_state.phi_min = st.number_input(
                "φ min",
                value=st.session_state.phi_min,
                max_value=st.session_state.phi_max,
                key="phi_min_input"
            )

        with col_p3:
            st.session_state.phi_max = st.number_input(
                "φ max",
                value=st.session_state.phi_max,
                min_value=st.session_state.phi_min,
                key="phi_max_input"
            )

        selected_recon_method = st.selectbox('Reconstruction Method',
                                         list(RECON_METHOD_MAPPING.keys()))
        st.session_state.recon_method = RECON_METHOD_MAPPING[selected_recon_method]
        beam_geometry = st.selectbox('Beam Geometry', ['Angular', 'Columnated'])
        st.session_state.is_beam_angular = beam_geometry == 'Angular'
        st.session_state.beam_width = st.number_input("Manual Width(deg/cm)",
                                             value=0.1,
                                             min_value=0.,
                                             disabled=True)
        st.session_state.radius = st.number_input("Arm distance",
                                              value=17.,
                                              min_value=0.,
                                              disabled=True)
        selected_width = st.selectbox("Width Mode",
                                  list(WIDTH_MODE_MAPPING.keys()),
                                  index=1,
                                  disabled=True
                                  )
        st.session_state.width_mode = WIDTH_MODE_MAPPING[selected_width]
        selected_filter = st.selectbox("Filter Mode",
                                    list(FILTER_MODE_MAPPING.keys()))
        st.session_state.filter_mode = FILTER_MODE_MAPPING[selected_filter]
    
    
    # hash check for update
    input_params = CTInputParams.from_session_state(st)
    if input_params.detect_hash_change(st):
        update_derived_state()
        initialise_playback_session_state()

#TODO:#play, pause, skip, stop
if st.button("▶️ Generate Reconstruction"):
    #st.session_state.isPlaying = True
    
    st_back_propagation()
    
    #st.session_state.isPlaying = False


if "recon_array" in st.session_state:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(st.session_state.recon_array, 
              #cmap='Grays',
              cmap=cmap(st.session_state.white_bg),
              origin='lower')
    ax.axis('off')
    st.pyplot(fig)

# debug
st.write(dict(st.session_state))

