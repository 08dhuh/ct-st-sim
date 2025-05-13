param_default = {
    "x_min": -5,
    "x_max": 5,
    "y_min": -5,
    "y_max": 5,
    "grid_count": 250,
    "object_rows": [[-2., -1.,  0.5],
                     [3., -2.,  1.],
                     [0.,  2.,  2.]],

    "white_bg":True,
    'is_beam_angular':True,
    'width_mode': 'static',
    'filter_mode': 'default',
    'recon_method': 'bp'

}

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