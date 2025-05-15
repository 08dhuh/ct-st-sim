DEFAULT_PARAM_VALUES = {
    "x_min": -5,
    "x_max": 5,
    "y_min": -5,
    "y_max": 5,
    "grid_count": 250,
    "theta_count": 100,
    "theta_max": 360.0,
    "theta_min": 0.0,
    "phi_count": 50,
    "phi_max": 18.0,
    "phi_min": -18.0,
    "object_rows": [[-2., -1.,  0.5],
                     [3., -2.,  1.],
                     [0.,  2.,  2.]], #TODO:rename

    "white_bg":True,
    'is_beam_angular':True,
    'width_mode': 'static',
    'filter_mode': 'default',
    'recon_method': 'bp',
    'beam_width': 1.,
    'radius': 17.

}


USER_INPUT_PARAM_NAMES = [
    "theta_count", "theta_max", "theta_min",
    "phi_count", "phi_max", "phi_min",
    "x_min", "x_max", "y_min", "y_max",
    "grid_count", "is_beam_angular",
    "radius", "beam_width", "width_mode", "filter_mode"
]

BP_PARAM_NAMES= [
    "theta_space",
    "phi_space",
    "object_rows",
    "radius",
    'filter_mode',
    "grid_count",
    "is_beam_angular",
    "beam_width",
    "width_mode",
    "xy_mesh"
]

WIDTH_MODE_MAPPING = {
    "Manual": "manual",
    "Auto-Static": "static",
    "Auto-Variable": "var",
}

FILTER_MODE_MAPPING = {
    "Default": "default",
    "Ramp Filter": "ramp",
    "Soft Tissue": "soft",
    "Bone": "bone",
}

RECON_METHOD_MAPPING = {
    'Back Propagation':'bp', 
    'Radon Transform':'rt'
}