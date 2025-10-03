import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from ct_core import ct_sim_main as ct
from config import WIDTH_MODE_MAPPING, FILTER_MODE_MAPPING, RECON_METHOD_MAPPING
from session_helpers import CTInputParams, st_back_propagation, st_radon_transform, initialise_session_state, update_derived_state, cmap, apply_cutoff_filter
from vid_utils import plot_array_px

import time

st.set_page_config(
    page_title="CT Scanner Simulator",
    page_icon="🩻",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    
           /* Remove the entire Streamlit header */
            header[data-testid="stHeader"] {
            display: none !important;
            }
            /* header */
            .st-emotion-cache-1w723zb {
            padding-top: 0rem !important;
            }

            /* Optional: remove extra space left behind */
            .main .block-container {
            padding-top: 0rem !important;
            }
           
           /* Remove blank space at the center canvas */ 
           .st-emotion-cache-z5fcl4 {
               position: relative;
               top: -62px;
               }
           
           /* Make the toolbar transparent and the content below it clickable */ 
           .st-emotion-cache-18ni7ap {
               pointer-events: none;
               background: rgb(255 255 255 / 0%)
               }
           .st-emotion-cache-zq5wmm {
               pointer-events: auto;
               background: rgb(255 255 255);
               border-radius: 5px;
               }
            /* Hide the entire sidebar header (logo spacer + collapse button) */
            /*[data-testid="stSidebarHeader"] {
                display: none !important;
            }*/
            /* Reduce the sidebar padding */
            .st-emotion-cache-1xgtwnd {
                padding: 0rem !important;
                }
                }

            
    </style>
    """, unsafe_allow_html=True)

# session state initialisation
initialise_session_state()


plot_dynamic = st.empty()
plot_static = st.container()


filter_cutoff = st.session_state.get("filter_cutoff", 0.0)
soft_cutoff = st.session_state.get("soft_cutoff", False)

play_step = 4
delay = 0.01


with st.sidebar:
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        if st.button(" Run Scan "):

            match st.session_state.recon_method:
                case 'bp':
                    st_back_propagation()
                case 'rt':
                    # st_radon_transform()
                    # TODO:
                    raise Exception(
                        "💀 Sorry! This feature is not implemented yet! Please hit F5 and try another💀")
                case _:
                    raise ValueError('Invalid recon method detected.')
            st.session_state.viewport_data = st.session_state.recon_array
            st.session_state.viewport_data_series = st.session_state.recon_array_series
            st.session_state.update({'play_triggered': True})

    with c2:
        is_playing = st.session_state.get("is_playing", False)
        play_triggered = st.session_state.get("play_triggered", False)
        disabled = len(st.session_state.get("viewport_data_series", [])) == 0
        if play_triggered:
            if st.button("⏹️ Stop", help="Stop animation early"):
                st.session_state.stop_requested = True
        else:
            if st.button("▶️ Replay", disabled=disabled, help="Start animation"):
                st.session_state.play_triggered = True

                st.rerun()

    st.markdown("# Filtering")

    with st.expander("Cutoff Threshold", expanded=True):
        filter_cutoff = st.session_state.get("filter_cutoff", 0.0)
        soft_cutoff = st.session_state.get("soft_cutoff", False)
        st.slider("Signal Cut-Off Threshold", 0.0,
                  1.0, key="filter_cutoff", step=0.01, label_visibility="hidden")
        st.checkbox("Soft Cutoff", key="soft_cutoff")
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
        st.session_state.white_bg = st.checkbox('White Background', value=True)
        obj_tuple = tuple(tuple(obj) for obj in st.session_state.object_rows)
        obj_masks = ct.run_objects_generation(params_tuple=obj_tuple,
                                              gridnum=st.session_state.grid_count,
                                              x_range=st.session_state.x_range,
                                              y_range=st.session_state.y_range,
                                              white_bg=True,
                                              )

        st.markdown('##### Preview')
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(obj_masks, cmap=cmap(not st.session_state.white_bg),
                  origin='lower', vmin=0, vmax=1)
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
        beam_geometry = st.selectbox(
            'Beam Geometry', ['Angular', 'Columnated'])
        st.session_state.is_beam_angular = beam_geometry == 'Angular'
        st.session_state.beam_width = st.number_input("Manual Width(deg/cm)",
                                                      value=0.1,
                                                      min_value=0.,
                                                      disabled=False)
        st.session_state.radius = st.number_input("Arm distance",
                                                  value=17.,
                                                  min_value=0.,
                                                  disabled=True)
        selected_width = st.selectbox("Width Mode",
                                      list(WIDTH_MODE_MAPPING.keys()),
                                      index=1,
                                      disabled=False
                                      )
        st.session_state.width_mode = WIDTH_MODE_MAPPING[selected_width]
        selected_filter = st.selectbox("Filter Mode",
                                       list(FILTER_MODE_MAPPING.keys()))
        st.session_state.filter_mode = FILTER_MODE_MAPPING[selected_filter]

    # hash check for update
    input_params = CTInputParams.from_session_state(st)
    if input_params.detect_hash_change(st):
        update_derived_state()


if st.session_state.get("play_triggered", False):
    st.session_state.is_playing = True
    plot_static.write("")
    for i in range(0, len(st.session_state.viewport_data_series), play_step):
        if st.session_state.get("stop_requested", False):
            break

        frame= st.session_state.viewport_data_series[i]
        
        fig = plot_array_px(frame, white_bg=st.session_state.white_bg)
        fig.update_layout(transition_duration=0)
        plot_dynamic.plotly_chart(fig, use_container_width=False)
        time.sleep(delay)

    filtered_data = apply_cutoff_filter(
        st.session_state.viewport_data,
        threshold=st.session_state.get("filter_cutoff", 0.0),
        soft=st.session_state.get("soft_cutoff", False)
    )
    x_vals = st.session_state.xy_mesh[0][0]
    y_vals = st.session_state.xy_mesh[1][:, 0]
    fig = plot_array_px(
        filtered_data,
        white_bg=st.session_state.white_bg,
        x_vals=x_vals,
        y_vals=y_vals)
    plot_dynamic.plotly_chart(fig, use_container_width=False, key='final')

    st.session_state.play_triggered = False
    st.session_state.is_playing = False
    st.session_state.stop_requested = False
    st.rerun()


if "viewport_data" in st.session_state:
    x_vals = st.session_state.xy_mesh[0][0]
    y_vals = st.session_state.xy_mesh[1][:, 0]
    if not st.session_state.get("play_triggered", False):
        filtered_data = apply_cutoff_filter(
            st.session_state.viewport_data,
            threshold=filter_cutoff,
            soft=soft_cutoff
        )
        fig = plot_array_px(filtered_data,
                            white_bg=st.session_state.white_bg,
                            x_vals=x_vals,
                            y_vals=y_vals)
        plot_static.plotly_chart(fig, use_container_width=False)


st.write('\n')

st.markdown("""
**Instructions**

- Expand the sidebar using the `>` icon in the top-left corner if it's collapsed.
- Will import your original documentation and add more controls later
            """)
# debug
# st.write(dict(st.session_state))
