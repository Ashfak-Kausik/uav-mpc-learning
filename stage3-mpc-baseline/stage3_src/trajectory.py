"""
This is the reference trajectory definitions for the MPC (Model Predictive Controll).

Each functions resturns position p_ref, velocity v_ref, and target Euler angles (which are zero for level flight- yaw can be specified per trajectory is needed).
The MPC consumes these as the full state reference 
x_ref = [p_ref, v_ref, 0, 0, yaw_ref].

The trajectory is sampled at the MPC's prediction timestep, dt_mpc.
"""

import numpy as np

def hover_reference(t_query, p_hover=np.array([0.0, 0.0, 1.0])):
    """
    Stationary hover at p_hover.

    Args:
        t_query: array of querry times (s); shape (N,)
        p_hover: (3,) hover position

    Returns:
        x_ref: (9, N) references state trajectory
    """
    N = len(t_query)
    x_ref = np.zeros((9, N))
    x_ref[0:3, :] = p_hover[:, None]                 # velocity orientation all zero 
    return x_ref 


def step_reference(t_query, t_step=1.0,
                   p_before=np.array([0.0, 0.0, 1.0]),
                   p_after=np.array([2.0, 0.0, 1.0])):
    """
    Hover at p_before until t_step, then hover at p_after.

    Args:
        t_query: array of querry times (s)
        t_step: step time (s)
        p_before, p_after: (3,) positions

    Returns
        x_ref: (9,N) reference state trajectory
    """
    N = len(t_query)
    x_ref = np.zeros((9,N))
    for i, t in enumerate(t_query):
        x_ref[0:3, i] = p_after if t >= t_step else p_before
    return x_ref 

def circle_reference(t_query, radius=1.5, period=6.0, z_alt=1.2,
                     t_ramp=2.0):
    """
    Circle in the xy-plane at constant altitude. Includes a smooth ramp-in
    over t_ramp seconds: position, velocity, and acceleration all start
    at zero and reach the full circle smoothly.

    Uses a cubic ramp factor s(t) = 3*tau^2 - 2*tau^3 (for tau = t/t_ramp)
    which has s(0)=0, s(t_ramp)=1, and s_dot(0)=s_dot(t_ramp)=0.

    Args:
        t_query : array of query times (s)
        radius  : final circle radius (m)
        period  : period of one revolution (s)
        z_alt   : altitude (m)
        t_ramp  : ramp duration (s)

    Returns:
        x_ref : (9, N) reference state trajectory
    """
    N = len(t_query)
    x_ref = np.zeros((9, N))
    omega = 2 * np.pi / period

    for i, t in enumerate(t_query):
        # Smooth ramp factor (cubic, zero-derivative at both ends)
        if t_ramp > 0 and t < t_ramp:
            tau = t / t_ramp
            s = 3 * tau**2 - 2 * tau**3
            sdot = (6 * tau - 6 * tau**2) / t_ramp
        else:
            s = 1.0
            sdot = 0.0

        c, s_om = np.cos(omega * t), np.sin(omega * t)

        # Full circle position and velocity (ungated)
        px_full = radius * c
        py_full = radius * s_om
        vx_full = -radius * omega * s_om
        vy_full = radius * omega * c

        # Apply the ramp: y = s * y_full, so dy/dt = sdot * y_full + s * dy_full/dt
        x_ref[0, i] = s * px_full
        x_ref[1, i] = s * py_full
        x_ref[2, i] = z_alt
        x_ref[3, i] = sdot * px_full + s * vx_full
        x_ref[4, i] = sdot * py_full + s * vy_full
        x_ref[5, i] = 0.0

    return x_ref
