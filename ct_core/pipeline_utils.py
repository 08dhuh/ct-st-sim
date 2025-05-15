import numpy as np


def generate_mesh_grid(x_range: tuple[float, float],
                       y_range: tuple[float, float],
                       num_grid: int):
    # TODO: fix
    return np.meshgrid(np.linspace(x_range[0], x_range[1], num_grid),
                       np.linspace(y_range[0], y_range[1], num_grid))


def sim_params_to_grid(params_tuple: tuple[tuple[float]],  # ((x1, y1, r1),(x2, y2, r2),...)
                       gridnum=250,
                       x_range=(-5, 5),
                       y_range=(-5, 5),
                       # plot=False,
                       white_bg=True) -> np.ndarray:
    # take the (x,y,r) parameters from the simulator and return a numpy array to visualise
    # set up x,y meshgrids based on a grid number and x and y ranges and then check if they are within the regions
    # defined by each circle
    # x1, y1, r1, x2, y2, r2, x3, y3, r3 = params_tuple
    x_mesh, y_mesh = generate_mesh_grid(
        x_range=x_range, y_range=y_range, num_grid=gridnum)
    output = np.zeros_like(x_mesh, dtype=int)
    for x, y, r in params_tuple:
        output |= (x_mesh - x) ** 2 + (y_mesh - y) ** 2 < r ** 2

    # output = (((x_mesh - x1) ** 2 + (y_mesh - y1) ** 2 < r1 ** 2)
    #           | ((x_mesh - x2) ** 2 + (y_mesh - y2) ** 2 < r2 ** 2)
    #           | ((x_mesh - x3) ** 2 + (y_mesh - y3) ** 2 < r3 ** 2)).astype(int)

    if not white_bg:
        output = 1 - output

    return output


def signal_readout_iter(obj_params: np.ndarray,
                        theta: float,
                        phi: float,
                        radius: float,
                        # width=False
                        ) -> bool:
    # take a single theta and phi value and check if it overlaps with the simulated objects. Returns either 0 or 1
    # x1, y1, r1, x2, y2, r2, x3, y3, r3 = obj_params
    def cond(x, y, r): return np.abs(np.sin(phi - theta) * (x - np.cos(theta)
                                                            * radius) + np.cos(phi - theta) * (y - np.sin(theta) * radius)) <= r

    return int(any([cond(x, y, r) for x, y, r in obj_params]))
#     nu_1 = np.sin(phi - theta) * (x1 - np.cos(theta) * radius) + np.cos(phi - theta) * (y1 - np.sin(theta) * radius)
#     nu_2 = np.sin(phi - theta) * (x2 - np.cos(theta) * radius) + np.cos(phi - theta) * (y2 - np.sin(theta) * radius)
#     nu_3 = np.sin(phi - theta) * (x3 - np.cos(theta) * radius) + np.cos(phi - theta) * (y3 - np.sin(theta) * radius)
#     if np.abs(nu_1) <= r1 or np.abs(nu_2) <= r2 or np.abs(nu_3) <= r3:
#         return 1
#     else:
#         return 0


def phi_width(phi: float,            
              

              # theta: float,
              phi_count: int,
              theta_count: int,
              phi_range: int,
              theta_range: int,
              radius: float,
              beam_width: float,
              width_mode: str,
              is_beam_angular: bool,
              ) -> tuple[float, float]:
    """
    three cases of width modes ['manual', 'static' , 'variable']
    """
    match width_mode.lower():
        case 'manual':
            if is_beam_angular:
                val = np.deg2rad(beam_width)
            else:
                val = beam_width
        case 'static':
            if is_beam_angular:
                val = phi_range / (2 * phi_count - 2)
            else:
                dtheta = min(phi_range / (phi_count - 1),
                             theta_range / (theta_count - 1))
                val = np.sin(dtheta / 2) * radius

        case 'variable':
            if is_beam_angular:
                val = phi_range / (2 * phi_count - 2)
            else:
                dtheta = theta_range / (theta_count - 1)
                val = np.sin((np.pi-dtheta)/2-phi)*np.sin(dtheta/2)*radius
    return val, val


def process_params(theta_tuple,
                   phi_tuple,
                   x_range,
                   y_range,
                   grid_count,
                   is_beam_angular: bool,
                   radius: float,
                   beam_width: float,
                   width_mode: str,
                   # filter_mode: str,
                   #    grid_min=100,
                   #    grid_max=700
                   ):
    """
    args

    theta_tuple,
                   phi_tuple,
                   x_range,
                   y_range,
                   grid_count,
                   is_beam_angular: bool,
                   radius: float,
                   width: float,
                   width_mode: str,



    returns
    theta_space,phi_space, xy_mesh, common_width_kwargs, strength, recon_array 

    """
    theta_count, theta_min, theta_max = theta_tuple
    phi_count, phi_min, phi_max = phi_tuple

    if abs(theta_max - 2 * np.pi) < 1e-4:
        theta_space = np.linspace(
            theta_min, theta_max, theta_count+1)[:-1]
    else:
        theta_space = np.linspace(
            theta_min, theta_max, theta_count)
    phi_space = np.linspace(phi_min, phi_max, phi_count)

    xy_mesh = generate_mesh_grid(x_range, y_range, grid_count)
    theta_range = theta_max - theta_min
    phi_range = phi_max - phi_min

    print(theta_range)

    if theta_range < 0:
        theta_range += 2 * np.pi
    if phi_range < 0:
        phi_range += 2 * np.pi

    # grid_count = np.clip(grid_count, grid_min, grid_max)
    grid_count = grid_count

    common_width_kwargs = extract_common_width_kwargs(
        is_beam_angular=is_beam_angular,
        beam_width=beam_width,
        phi_count=phi_tuple[0],
        theta_count=theta_tuple[0],
        phi_range=phi_range,
        theta_range=theta_range,
        radius=radius
    )
    # strength = _calculate_beam_strength_scaling(width_mode=width_mode,
    #                                             common_width_kwargs=common_width_kwargs)
    strength = 1
    recon_array = np.zeros((grid_count, grid_count))
    return theta_space, phi_space, xy_mesh, common_width_kwargs, strength, recon_array


def _calculate_beam_strength_scaling(width_mode: str, common_width_kwargs: dict) -> float:
    """
    """
    if width_mode == 'manual':
        auto_width = phi_width(
                               phi=0,
                               width_mode='static',
                               #theta=0,
                               **common_width_kwargs)[0]
        manual_width = phi_width(
                                 phi=0,
                                 width_mode='manual',
                                 #heta=0,
                                 **common_width_kwargs)[0]
        return auto_width/manual_width
    else:
        return 1


def extract_common_width_kwargs(is_beam_angular: bool,
                                beam_width: float,
                                phi_count: int,
                                theta_count: int,
                                phi_range: int,
                                theta_range: int,
                                radius: float):
    return {
        "is_beam_angular": is_beam_angular,
        "beam_width": beam_width,
        "phi_count": phi_count,
        "theta_count": theta_count,
        "phi_range": phi_range,
        "theta_range": theta_range,
        "radius": radius,
    }
