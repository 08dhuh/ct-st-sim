import numpy as np
from ct_core.pipeline_utils import generate_mesh_grid


def ray_id_vectorised(theta: np.ndarray,
           phi: np.ndarray,
           R: float,
           grid_count: int,
           is_beam_angular: bool,
           xy_mesh: tuple[np.ndarray, np.ndarray] | None = None,
           x_range: tuple[float, float] = (-5, 5),
           y_range: tuple[float, float] = (-5, 5),
           # mode="p",
           width: tuple[float, float] | float = 0.1) -> np.ndarray:

    if isinstance(width, (int, float, np.generic)):
        width_p = width_n = float(width)
    elif isinstance(width, tuple) and len(width) == 2:
        width_p, width_n = map(float, width)

    x, y = xy_mesh if xy_mesh is not None else generate_mesh_grid(
        x_range, y_range, num_grid=grid_count)
    x = x[None, None, :, :]
    y = y[None, None, :, :]

    theta = theta[:, None, None, None]
    phi = phi[None, :, None, None]

    epsilon = 1e-8

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi_theta = np.cos(phi - theta)
    sin_phi_theta = np.sin(phi - theta)
    tan_theta_phi = np.tan(theta - phi)
    tan_theta_phi = np.where(np.abs(tan_theta_phi) < epsilon, np.inf, tan_theta_phi)

    R_cos_theta =  R*cos_theta
    R_sin_theta = R*sin_theta

    dx = x - R_cos_theta
    dy = y - R_sin_theta

    d_angle = theta - phi

    if is_beam_angular:  # angular beam logic
        is_vertical = (np.abs(np.abs(d_angle) - np.pi / 2) <
                       width_p) | (np.abs(np.abs(d_angle) - 3 * np.pi / 2) < width_p)

        slope = np.where(is_vertical, dx / dy, dy/dx)
        upper_limit = np.where(is_vertical,
                               np.tan(-d_angle + width_p + np.pi / 2),
                               np.tan(d_angle + width_p))
        lower_limit = np.where(is_vertical,
                               np.tan(-d_angle - width_n - np.pi / 2),
                               np.tan(d_angle - width_n))

        ray_array = ((lower_limit <= slope) & (
            slope <= upper_limit)).astype(int)
    else:  # parallel beam logic
        is_vertical = np.abs((d_angle % (2 * np.pi) - np.pi/2) %
                             np.pi - np.pi/2) <= np.pi / 4
        is_positive_v = (np.abs(d_angle + np.pi / 2) <= np.pi /
                         2) | (np.abs(d_angle - 3 * np.pi / 2) <= np.pi / 2)
        is_positive_h = (np.abs(d_angle) <= np.pi /
                         2) | (np.abs(d_angle - 2 * np.pi) <= np.pi / 2)
        
        mask_vp = is_vertical & is_positive_v
        mask_vn = is_vertical & ~is_positive_v
        mask_hp = ~is_vertical & is_positive_h
        mask_hn = ~is_vertical & ~is_positive_h

        term1_v = (y - R_sin_theta - width_p * cos_phi_theta) / tan_theta_phi +\
            R_cos_theta + width_p * sin_phi_theta
        term2_v = (y - R_sin_theta + width_n * cos_phi_theta) / tan_theta_phi +\
            R_cos_theta- width_n * sin_phi_theta
        term1_h = tan_theta_phi * (x - R_cos_theta -
                                       width_p * sin_phi_theta) \
            + R * sin_theta + width_p * cos_phi_theta
        term2_h = tan_theta_phi * (
            x - R_cos_theta + width_n * sin_phi_theta) + \
            R_sin_theta - width_n * cos_phi_theta

        ray_array = np.zeros((theta.shape[0], phi.shape[1], grid_count, grid_count), dtype=int)


        ray_array += mask_vp * ((term1_v >= x) & (x >= term2_v))
        ray_array += mask_vn * ((term1_v <= x) & (x <= term2_v))

        ray_array += mask_hp * ((term1_h >= y) & (y >= term2_h))
        ray_array += mask_hn * ((term1_h <= y) & (y <= term2_h))
        
        ray_array = ray_array.astype(int)
    return ray_array


def ray_id_scalar(theta: float,
           phi: float,
           radius:float,
           grid_count:int,
           is_beam_angular: bool=True,
           xy_mesh: tuple[np.ndarray, np.ndarray] | None = None,
           x_range: tuple[float, float] = (-5, 5),
           y_range: tuple[float, float] = (-5, 5),
           # mode="p",
           beam_width: tuple[float, float] | float = 0.1) -> np.ndarray:

    x, y = xy_mesh if xy_mesh is not None else generate_mesh_grid(
        x_range, y_range, num_grid=grid_count)

    if isinstance(beam_width, (int, float, np.generic)):
        width_p = width_n = float(beam_width)
    elif isinstance(beam_width, tuple) and len(beam_width) == 2:
        width_p, width_n = map(float, beam_width)
    
    epsilon = 1e-8
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi_theta = np.cos(phi - theta)
    sin_phi_theta = np.sin(phi - theta)
    tan_theta_phi = np.tan(theta - phi)
    tan_theta_phi = np.where(np.abs(tan_theta_phi) < epsilon, np.inf, tan_theta_phi)

    R_cos_theta =  radius*cos_theta
    R_sin_theta = radius*sin_theta

    dx = x - R_cos_theta
    dy = y - R_sin_theta

    d_angle = theta - phi

    

    if is_beam_angular:  # angular beam logic
        is_vertical = abs(abs(d_angle) - np.pi /
                          2) < width_p or abs(abs(d_angle) - 3 * np.pi / 2) < width_p

        if is_vertical:
            slope = dx / dy
            upper_limit = np.tan(-d_angle + width_p + np.pi / 2)
            lower_limit = np.tan(-d_angle - width_n - np.pi / 2)
        else:
            slope = dy/dx
            upper_limit = np.tan(d_angle + width_p)
            lower_limit = np.tan(d_angle - width_n)

        ray_array = (
            (slope <= upper_limit) & (lower_limit <= slope)).astype(int)
    else:  # parallel beam logic
        is_vertical = abs((d_angle % (2*np.pi) - np.pi/2) % np.pi - np.pi/2) <= np.pi / 4
        is_positive_v = abs(d_angle + np.pi/2) <= np.pi/2 or abs(d_angle - 3*np.pi/2) <= np.pi/2
        is_positive_h = abs(d_angle) <= np.pi/2 or abs(d_angle - 2*np.pi) <= np.pi/2

        term1_v = (y - R_sin_theta - width_p*cos_phi_theta)/tan_theta_phi + R_cos_theta + width_p*sin_phi_theta
        term2_v = (y - R_sin_theta + width_n *cos_phi_theta)/tan_theta_phi + R_cos_theta - width_n*sin_phi_theta

        term1_h = tan_theta_phi*(x - R_cos_theta - width_p*sin_phi_theta) + R_sin_theta + width_p* cos_phi_theta
        term2_h = tan_theta_phi*(x - R_cos_theta + width_n*sin_phi_theta) + R_sin_theta - width_n *cos_phi_theta
        ray_array = np.zeros_like(x, dtype=int)
        if is_vertical:
            if is_positive_v:
                ray_array = ((term1_v >= x) & (x>=term2_v)).astype(int)
            else:
                ray_array = ((term1_v <= x) & (x<=term2_v)).astype(int)
        else:
            if is_positive_h:
                ray_array = ((term1_h >= y) & (y>=term2_h)).astype(int)
            else:
                ray_array = ((term1_h <= y) & (y<=term2_h)).astype(int)

    return ray_array




def compute_signal_mask(params_tuple,
                        theta_space: np.ndarray,
                        phi_space: np.ndarray,
                        radius: float) -> np.ndarray:
    #binary mask indicating which rays intersect any object

    x1, y1, r1, x2, y2, r2, x3, y3, r3 = params_tuple
    theta = theta_space[:, None]
    phi = phi_space[None, :]

    dx = lambda x: x - np.cos(theta) * radius
    dy = lambda y: y - np.sin(theta) * radius

    def in_object(x, y, r):
        nu = np.sin(phi - theta) * dx(x) + np.cos(phi - theta) * dy(y)
        return (np.abs(nu) <= r)

    mask = in_object(x1, y1, r1) | in_object(x2, y2, r2) | in_object(x3, y3, r3)
    return mask.astype(np.float32)

def back_propagation_from_rays(ray_array: np.ndarray, #masked
                               strength: float,
                               theta_count: int) -> np.ndarray:
    scaled_rays = ray_array * (strength / theta_count)
    return scaled_rays.cumsum(axis=0).cumsum(axis=1).astype(np.float32)


