import streamlit as st

import matplotlib.pyplot as plt

from ct_core import ct_sim_main as ct
from ct_core.config import param_default, WIDTH_MODE_MAPPING, FILTER_MODE_MAPPING,RECON_METHOD_MAPPING

#TODO: add errors
#TODO: add hash sum checks

# session state initialisation
for key, value in param_default.items():
    st.session_state.setdefault(key, value)

st.session_state.setdefault(
    "x_range", (st.session_state.x_min, st.session_state.x_max))
st.session_state.setdefault(
    "y_range", (st.session_state.y_min, st.session_state.y_max))


st.title('CT Scanner Simulator')

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
""")

with st.sidebar:
    # sidebar
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
        obj_tuple = tuple(tuple(obj) for obj in st.session_state.object_rows)
        obj_masks = ct.run_objects_generation(params_tuple=obj_tuple,
                                              gridnum=st.session_state.grid_count,
                                              x_range=st.session_state.x_range,
                                              y_range=st.session_state.y_range,
                                              white_bg=st.session_state.white_bg
                                              )
        st.markdown('##### Preview')
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(obj_masks, cmap='gray', origin='lower')
        ax.axis('off')
        st.pyplot(fig)

        # ct.run_objects_generation
        # write code from here

    with st.expander("Image Grid Parameters", expanded=True):
        st.markdown("### X Range")
        col_x1, col_x2 = st.columns(2)
        col_y1, col_y2 = st.columns(2)

        with col_x1:
            st.session_state.x_min = st.number_input("X min",
                                                     value=st.session_state.x_min,
                                                     max_value=st.session_state.x_max - 1,
                                                     key='x_min_input')
        with col_x2:
            st.session_state.x_max = st.number_input("X max",
                                                     value=st.session_state.x_max,
                                                     min_value=st.session_state.x_min + 1,
                                                     key='x_max_input')
        st.markdown("### Y Range")
        with col_y1:
            st.session_state.y_min = st.number_input("Y min",
                                                     value=st.session_state.y_min,
                                                     max_value=st.session_state.y_max - 1,
                                                     key='y_min_input')
        with col_y2:
            st.session_state.y_max = st.number_input("Y max",
                                                     value=st.session_state.y_max,
                                                     min_value=st.session_state.y_min + 1,
                                                     key='y_max_input')
        st.session_state.grid_count = st.number_input(
            label="grid count",
            min_value=100,
            max_value=700,
            value=int(st.session_state.grid_count),
            step=50)

    selected_recon_method= st.selectbox('Reconstruction Method',
                                list(RECON_METHOD_MAPPING.keys()))
    st.session_state.recon_method = RECON_METHOD_MAPPING[selected_recon_method]
    st.session_state.white_bg = st.checkbox('White Background')
    beam_geometry = st.selectbox('Beam Geometry', ['Angular', 'Columnated'])
    st.session_state.is_beam_angular = beam_geometry == 'Angular'

    selected_width = st.selectbox("Width Mode",
                              list(WIDTH_MODE_MAPPING.keys())
                              )
    st.session_state.width_mode = WIDTH_MODE_MAPPING[selected_width]
    selected_filter= st.selectbox("Filter Mode",
                               list(FILTER_MODE_MAPPING.keys()))
    st.session_state.filter_mode = FILTER_MODE_MAPPING[selected_filter]

# viewports
fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow()

#debug
st.write([(key,value) for key, value in st.session_state.items()])


def main():
    print("Hello from ct-st-sim!")


if __name__ == "__main__":
    main()
