import os
import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import interp1d

import bee_simulator
import central_complex
import cx_rate
import cx_basic

default_acc = 0.15  # A good value because keeps speed under 1
default_drag = 0.15

DATA_PATH = "data"


class CXLogger(object):
    """Class to store logs in of central complex cell activations."""

    def __init__(self, T_outbound, T_inbound, cx=None):
        """Initialise log as many zerod out numpy arrays."""
        self.T_outbound = T_outbound
        self.T_inbound = T_inbound
        T = T_outbound + T_inbound
        self.T = T
        self.cx = cx
        if issubclass(cx.__class__, cx_basic.CXBasic):
            self.tl2 = np.empty([1, T])
            self.cl1 = np.empty([1, T])
        else:
            self.tl2 = np.empty([central_complex.N_TL2, T])
            self.cl1 = np.empty([central_complex.N_CL1, T])
        self.tb1 = np.empty([central_complex.N_TB1, T])
        self.tn1 = np.empty([central_complex.N_TN1, T])
        self.tn2 = np.empty([central_complex.N_TN2, T])
        self.memory = np.empty([central_complex.N_CPU4, T])
        self.cpu4 = np.empty([central_complex.N_CPU4, T])
        self.cpu1 = np.empty([
            central_complex.N_CPU1A + central_complex.N_CPU1B, T
        ])
        self.motor = np.empty(T)

    def update_log(self, t, tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1=np.nan,
                   motor=np.nan):
        """Add the latest value to each cell type."""
        self.tl2[:, t] = tl2
        self.cl1[:, t] = cl1
        self.tb1[:, t] = tb1
        self.tn1[:, t] = tn1
        self.tn2[:, t] = tn2
        self.memory[:, t] = memory
        self.cpu4[:, t] = cpu4
        self.cpu1[:, t] = cpu1
        self.motor[t] = motor

    def __add__(self, other):
        """Combine two logs into one big one (normally outbound and
        inbound)."""
        combined = CXLogger(T_outbound=self.T, T_inbound=other.T-1, cx=self.cx)
        combined.tl2[:, :self.T] = self.tl2
        combined.cl1[:, :self.T] = self.cl1
        combined.tb1[:, :self.T] = self.tb1
        combined.tn1[:, :self.T] = self.tn1
        combined.tn2[:, :self.T] = self.tn2
        combined.memory[:, :self.T] = self.memory
        combined.cpu4[:, :self.T] = self.cpu4
        combined.cpu1[:, :self.T] = self.cpu1
        combined.motor[:self.T] = self.motor

        # Here we skip the first element of inbound as duplicate (clumsy
        # coding, think of fix)
        combined.tl2[:, self.T:] = other.tl2[:, 1:]
        combined.cl1[:, self.T:] = other.cl1[:, 1:]
        combined.tb1[:, self.T:] = other.tb1[:, 1:]
        combined.tn1[:, self.T:] = other.tn1[:, 1:]
        combined.tn2[:, self.T:] = other.tn2[:, 1:]
        combined.memory[:, self.T:] = other.memory[:, 1:]
        combined.cpu4[:, self.T:] = other.cpu4[:, 1:]
        combined.cpu1[:, self.T:] = other.cpu1[:, 1:]
        combined.motor[self.T:] = other.motor[1:]
        return combined


def generate_route(T=1500, mean_acc=default_acc, drag=default_drag,
                   kappa=100.0, max_acc=default_acc, min_acc=0.0,
                   vary_speed=False):
    """Generate a random outbound route using bee_simulator physics.
    The rotations are drawn randomly from a von mises distribution and smoothed
    to ensure the agent makes more natural turns."""
    # Generate random turns
    mu = 0.0
    vm = np.random.vonmises(mu, kappa, T)
    rotation = lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    # Randomly sample some points within acceptable acceleration and
    # interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = T / 50
        else:
            num_key_speeds = 4
        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros([T, 2])

    for t in range(1, T):
        headings[t], velocity[t, :] = bee_simulator.get_next_state(
            heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
    return headings, velocity


def update_cells(heading, velocity, tb1, memory, cx, filtered_steps=0.0):
    """Generate activity for all cells, based on previous activity and current
    motion."""
    # Compass
    tl2 = cx.tl2_output(heading)
    cl1 = cx.cl1_output(tl2)
    tb1 = cx.tb1_output(cl1, tb1)

    # Speed
    flow = cx.get_flow(heading, velocity, filtered_steps)
    tn1 = cx.tn1_output(flow)
    tn2 = cx.tn2_output(flow)

    # Update memory for distance just travelled
    memory = cx.cpu4_update(memory, tb1, tn1, tn2)
    cpu4 = cx.cpu4_output(memory)

    # Steer based on memory and direction
    cpu1 = cx.cpu1_output(tb1, cpu4)
    motor = cx.motor_output(cpu1)
    return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor


def generate_memory(headings, velocity, cx, bump_shift=0.0, filtered_steps=0.0,
                    logging=False):
    """For an outbound route, generate all the cell activity."""
    T = len(headings)

    if logging:
        cx_log = CXLogger(T, 0, cx)

    # Initialise TB and memory
    tb1 = np.zeros(central_complex.N_TB1)
    memory = 0.5 * np.ones(central_complex.N_CPU4)

    for t in range(T):
        tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor = update_cells(
            heading=headings[t], velocity=velocity[t], tb1=tb1, memory=memory,
            cx=cx, filtered_steps=filtered_steps)
        if logging:
            cx_log.update_log(t, tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1,
                              motor)

    if logging:
        return cx_log
    else:
        return tl2, cl1, tb1, tn1, tn2, memory, cpu4


def homing(T, tb1, memory, cx, acceleration=default_acc, drag=default_drag,
           current_heading=0.0, current_velocity=np.array([0.0, 0.0]),
           turn_sharpness=1.0, logging=True, bump_shift=0.0,
           filtered_steps=0.0):
    """Based on current state, return home. First is duplicate"""
    headings = np.empty(T + 1)
    headings[0] = current_heading
    velocity = np.empty([T + 1, 2])
    velocity[0, :] = current_velocity

    if logging:
        cx_log = CXLogger(0, T + 1, cx)
    else:
        cx_log = None

    for t in range(1, T + 1):
        r = headings[t - 1] - headings[t - 2]
        r = (r + np.pi) % (2 * np.pi) - np.pi
        tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor = update_cells(
            heading=headings[t - 1] + np.sign(r) * bump_shift,  # Remove sign to use proportionate shift
            velocity=velocity[t - 1],
            tb1=tb1,
            memory=memory,
            cx=cx,
            filtered_steps=filtered_steps)
        if logging:
            cx_log.update_log(t, tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1,
                              motor)
        rotation = turn_sharpness * motor

        headings[t], velocity[t, :] = bee_simulator.get_next_state(
            headings[t - 1], velocity[t - 1, :], rotation, acceleration, drag)
    return headings, velocity, cx_log


def run_trial(route=None, T_outbound=1500, T_inbound=1500, acc_out=default_acc,
              acc_in=0.1, noise=0.1, weight_noise=0.0, vary_speed=True,
              cx=None, cx_class=cx_rate.CXRate, logging=True, random_homing=False, bump_shift=0.0,
              filtered_steps=0.0, drag=default_drag, tn_prefs=np.pi/4.0):
    """Generate outbound and inbound route and store results.

    Arguments:
    bump_shift refers to TB1 'pre-emting' activity in the direction of
    motion."""
    # First generate or load an outbound route.
    if route:
        h_out, v_out = route
    else:
        h_out, v_out = generate_route(T=T_outbound, mean_acc=acc_out, drag=drag,
                                      max_acc=acc_out, vary_speed=vary_speed)

    # Use generated route to update cells in central complex.
    if cx is None:
        cx = cx_class(noise=noise, weight_noise=weight_noise,
                      tn_prefs=tn_prefs)

    if logging:
        log_out = generate_memory(h_out, v_out, cx, logging=True,
                                  bump_shift=bump_shift)
        tb1 = log_out.tb1[:, -1]
        memory = log_out.memory[:, -1]
    else:
        tl2, cl1, tb1, tn1, tn2, memory, cpu4 = generate_memory(
            headings=h_out, velocity=v_out, cx=cx, logging=logging,
            filtered_steps=filtered_steps, bump_shift=bump_shift)

    cpu4_snapshot = memory.copy()
    # Start homing and store headings, velocity and cell activity.
    if random_homing:
        # T_inbound+1 to take into account duplicate first value on normal runs
        h_in, v_in = generate_route(T=T_inbound+1, mean_acc=acc_in, drag=drag,
                                    max_acc=acc_in, vary_speed=vary_speed)
        if logging:
            log_in = generate_memory(headings=h_in, velocity=v_in, cx=cx,
                                     filtered_steps=filtered_steps,
                                     bump_shift=bump_shift, logging=True)
    else:
        h_in, v_in, log_in = homing(
            T=T_inbound, tb1=tb1, memory=memory, cx=cx,
            acceleration=acc_in, current_heading=h_out[-1],
            current_velocity=v_out[-1], logging=logging, bump_shift=bump_shift,
            filtered_steps=filtered_steps, drag=drag)
    h = np.hstack([h_out, h_in])
    v = np.vstack([v_out, v_in])

    if logging:
        log = log_out + log_in
    else:
        log = None
    return h, v, log, cpu4_snapshot


def load_route(filename='route.npz'):
    """Load a pre-traversed route."""
    with np.load(os.path.join(DATA_PATH, filename)) as data:
        h = data['h']
        v = data['v']
        T_outbound = data['T_outbound']
        T_inbound = data['T_inbound']
        cx_log = CXLogger(T_outbound=T_outbound, T_inbound=T_inbound)
        cx_log.tl2 = data['tl2']
        cx_log.cl1 = data['cl1']
        cx_log.tb1 = data['tb1']
        cx_log.tn1 = data['tn1']
        cx_log.tn2 = data['tn2']
        cx_log.memory = data['memory']
        cx_log.cpu4 = data['cpu4']
        cx_log.cpu1 = data['cpu1']
        cx_log.motor = data['motor']
    return h, v, cx_log


def save_route(h, v, cx_log, filename='route.npz'):
    """Save current data to a compressed numpy file (for generating an
    animation)."""
    np.savez_compressed(os.path.join(DATA_PATH, filename),
                        h=h,
                        v=v,
                        T_outbound=cx_log.T_outbound,
                        T_inbound=cx_log.T_inbound,
                        tl2=cx_log.tl2,
                        cl1=cx_log.cl1,
                        tb1=cx_log.tb1,
                        tn1=cx_log.tn1,
                        tn2=cx_log.tn2,
                        memory=cx_log.memory,
                        cpu4=cx_log.cpu4,
                        cpu1=cx_log.cpu1,
                        motor=cx_log.motor)


def generate_filename(T_outbound, T_inbound, noise, N, **kwargs):
    filename = 'out{0}_in{1}_noise{2}_N{3}'.format(str(T_outbound),
                                                str(T_inbound),
                                                str(noise),
                                                str(N))
    for k, v in kwargs.iteritems():
        filename += '_' + k + str(v)
    return filename + '.npz'


def load_dataset(T_outbound, T_inbound, noise, N, **kwargs):
    filename = generate_filename(T_outbound, T_inbound, noise, N,
                                 **kwargs)
    with np.load(os.path.join(DATA_PATH, filename)) as data:
        H = data['H']
        V = data['V']
        cpu4_snapshot = data['cpu4_snapshot']
    return H, V, cpu4_snapshot


def save_dataset(H, V, cpu4_snapshot, T_outbound, T_inbound, noise, N,
                 **kwargs):
    filename = generate_filename(T_outbound, T_inbound, noise, N,
                                 **kwargs)
    np.savez(os.path.join(DATA_PATH, filename), H=H, V=V,
             cpu4_snapshot=cpu4_snapshot)


def generate_dataset(T_outbound=1500, T_inbound=1500, noise=0.1, N=1000,
                     save=True, **kwargs):
    try:
        H, V, cpu4_snapshot = load_dataset(T_outbound, T_inbound, noise, N,
                                           **kwargs)
    except:
        T = T_outbound + T_inbound
        H = np.empty([N, T+1])
        V = np.empty([N, T+1, 2])  # TODO(tomish) why is this shape larger?
        cpu4_snapshot = np.empty([N, central_complex.N_CPU4])
        for i in range(N):
            H[i, :], V[i, :, :], _, cpu4_snapshot[i, :] = run_trial(
                    T_outbound=T_outbound,
                    T_inbound=T_inbound,
                    noise=noise,
                    logging=False,
                    **kwargs)
        if save:
            save_dataset(H, V, cpu4_snapshot, T_outbound, T_inbound, noise, N,
                         **kwargs)
    return H, V, cpu4_snapshot
