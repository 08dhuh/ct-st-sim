import plotly.express as px
import numpy as np
from session_helpers import cmap

def plot_array_px(array: np.ndarray, *, 
                  white_bg:bool=True, 
                  zoom:bool=True,
                  x_vals:np.ndarray | None = None,
                  y_vals:np.ndarray | None = None):
    fig = px.imshow(
        array,
        origin='lower',
        x=x_vals,
        y=y_vals,
        zmin=0,
        zmax=1,
        color_continuous_scale=cmap(not white_bg)
    )
    fig.update_layout(
        coloraxis_showscale=False,
        dragmode='zoom' if zoom else False,
        margin=dict(l=0, r=0, t=0, b=0),
        width=700,
        height=700,
        xaxis_title=None,
        yaxis_title=None
    )
    return fig
