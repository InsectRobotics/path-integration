import numpy as np
import scipy as sp

n_sensors = 20
directions = np.linspace(-np.pi, np.pi, n_sensors, endpoint=False)
D = np.column_stack((np.cos(directions), np.sin(directions),
                     np.zeros(n_sensors)))


def image_motion_flow(T, R, D):
    """Calculate optic flow based on movement."""
    P = T - (D.T * np.dot(D, T)).T
    P -= np.cross(R, D)
    return P


def rotary_flow(D, A):
    """Counterclockwise rotation."""
    return -np.cross(D, A)


def translatory_flow(D, A):
    return np.cross(np.cross(D, A), D)


def linear_range_model(U, P, w=1.0, n=0.0):
    """Eq 5 in Franz & Krapp"""
    return np.sum(w * (np.sum(U * P, axis=1) + n))


def tn_axes(heading):
    return np.array([[np.sin(heading - np.pi / 4.0),
                      np.cos(heading - np.pi / 4.0), 0],
                     [np.sin(heading + np.pi / 4.0),
                      np.cos(heading + np.pi / 4.0), 0]])


def get_flow2(heading, velocity):
    """This is the longwinded version that does all the flow calculations,
    piece by piece. It can be refactored down to flow2() so use that for
    performance benefit."""
    T = np.array([velocity[0], velocity[1], 0.0])  # We are keeping
    R = np.array([0.0, 0.0, 0.0])
    P = image_motion_flow(T, R, D)
    a = tn_axes(heading)

    U_TN_1 = translatory_flow(D, a[0])
    U_TN_2 = translatory_flow(D, a[1])

    lr_1 = linear_range_model(U_TN_1, P, w=1.0/10.0)
    lr_2 = linear_range_model(U_TN_2, P, w=1.0/10.0)
    return np.array([lr_1, lr_2])


def get_flow(heading, velocity, pref_angle=np.pi/4):
    A = np.array([[np.sin(heading - pref_angle),
                   np.cos(heading - pref_angle)],
                  [np.sin(heading + pref_angle),
                   np.cos(heading + pref_angle)]])
    return np.dot(A, velocity)


def rotate(theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r + np.pi) % (2.0 * np.pi) - np.pi


def thrust(theta, acceleration):
    """Thrust vector from current heading and acceleration

    theta: clockwise radians around z-axis, where 0 is forward
    acceleration: float where max speed is ....?!?
    """
    return np.array([np.sin(theta), np.cos(theta)]) * acceleration


def get_next_state(heading, velocity, rotation, acceleration, drag=0.5):
    """Get new heading and velocity, based on relative rotation and
    acceleration and linear drag."""
    theta = rotate(heading, rotation)
    v = velocity + thrust(theta, acceleration)
    v -= drag * v
    return theta, v


def run_simulation(N_outbound=500, N_inbound=1000):
    N = N_outbound + N_inbound

    for i in range(1, N-1):
        headings[i], v[i, :], flow[i, :] = get_next_state(
                headings[i-1], v[i-1, :], r[i], a, drag=0.25)
