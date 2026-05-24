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

def circle_reference(t_query, radius=1.5, period=6.0, z_alt=1.2, t_ramp=2.0):
    """
    Circle in the xy-plane at constant altitude. Includes a linear ramp-in of the radius over t_ramp seconds.

    Args:
        t_query: array of querry times (s)
        radius: final circle radius (m)
        period: period of one revolution (s)
        z_alt: altitude (m)
        t_ramp: ramp duration (s)

    Returns:
        x_ref: (9,N) reference state trajectory
    """
    N = len(t_query)
    x_ref = np.zeros((9,N))
    omega = 2 * np.pi / period 
    for i, t in enumerate(t_query):
        r = radius * min(t /t_ramp, 1.0) if t_ramp > 0 else radius 
        rdot = (radius / t_ramp) if t < t_ramp else 0.0
        c, s = np.cos(omega * t) , np.sin(omega * t)
        x_ref[0, i] = r * c
        x_ref[1, i] = r * s
        x_ref[2, i] = z_alt
        x_ref[3, i] = rdot * c - r * omega * s
        x_ref[4, i] = rdot * s + r * omega * c
        x_ref[5, i] = 0.0
    return x_ref
