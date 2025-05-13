import numpy as np

def signal_readout_iter(params_tuple, theta, phi, radius, width=False) -> int:
    # take a single theta and phi value and check if it overlaps with the simulated objects. Returns either 0 or 1
    x1, y1, r1, x2, y2, r2, x3, y3, r3 = params_tuple
    nu_1 = np.sin(phi - theta) * (x1 - np.cos(theta) * radius) + np.cos(phi - theta) * (y1 - np.sin(theta) * radius)
    nu_2 = np.sin(phi - theta) * (x2 - np.cos(theta) * radius) + np.cos(phi - theta) * (y2 - np.sin(theta) * radius)
    nu_3 = np.sin(phi - theta) * (x3 - np.cos(theta) * radius) + np.cos(phi - theta) * (y3 - np.sin(theta) * radius)
    if np.abs(nu_1) <= r1 or np.abs(nu_2) <= r2 or np.abs(nu_3) <= r3:
        return 1
    else:
        return 0

def signal_array(params_tuple, radius, gridnum=500, theta_range=(0, 2 * np.pi),
                 phi_range=(-7 * np.pi / 90, 7 * np.pi / 90)) -> np.ndarray:
    # Perform a skew-radon transform on the simulated data and return the (pixelated) radon transform of the data
    # This is essentially a vectorised version of the signal algorithm, since the signal to radon conversion is trivial

    x1, y1, r1, x2, y2, r2, x3, y3, r3 = params_tuple
    theta_mesh, phi_mesh = np.meshgrid(np.linspace(theta_range[0], theta_range[1], gridnum),
                                       np.linspace(phi_range[0], phi_range[1], gridnum))
    nu_1 = np.sin(phi_mesh - theta_mesh) * (x1 - np.cos(theta_mesh) * radius) + np.cos(phi_mesh - theta_mesh) \
           * (y1 - np.sin(theta_mesh) * radius)
    nu_2 = np.sin(phi_mesh - theta_mesh) * (x2 - np.cos(theta_mesh) * radius) + np.cos(phi_mesh - theta_mesh) \
           * (y2 - np.sin(theta_mesh) * radius)
    nu_3 = np.sin(phi_mesh - theta_mesh) * (x3 - np.cos(theta_mesh) * radius) + np.cos(phi_mesh - theta_mesh) \
           * (y3 - np.sin(theta_mesh) * radius)
    output = ((np.abs(nu_1) <= r1) | (np.abs(nu_2) <= r2) | (np.abs(nu_3) <= r3)).astype(int)
    return output
