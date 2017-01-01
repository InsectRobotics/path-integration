"""A simple script to simulate central complex in terms of sinusoids for
holonomic movement."""

from central_complex import *
import numpy as np

N_COLUMNS = 8  # Number of columns
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class CXBasic(CX):
    """Implements basic CX model but:
    - noise free.
    - perfect sinusoids for TB1.
    - memory can update using inverse amplitudes for TB1 (backwards motion).
    - perfect memory decay relative to speed."""

    def __init__(self, **kwargs):
        # 45 degree offset for optic flow preference angle.
        super(CXBasic, self).__init__(**kwargs)

    def tl2_output(self, theta):
        """Dummy function."""
        return theta

    def cl1_output(self, theta):
        """Dummy function."""
        return theta

    def tb1_output(self, theta, tb1=None):
        """Sinusoidal response to solar compass."""
        return (1.0 + np.cos(np.pi + x + theta)) / 2.0

    def tn1_output(self, flow):
        """Linearly inverse sensitive to forwards and backwards motion."""
        return np.clip((1.0 - flow) / 2.0, 0, 1)

    def tn2_output(self, flow):
        """Linearly sensitive to forwards motion only."""
        return np.clip(flow, 0, 1)

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """Updates memory based on current TB1 and TN activity.
        Can think of this as summing sinusoid of TB1 onto sinusoid of CPU4.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)

        # Idealised setup, where we can negate the TB1 sinusoid
        # for memorising backwards motion
        mem_update = (0.5 - tn1.reshape(2, 1)) * (1.0 - tb1)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_update -= 0.5 * (0.5 - tn1.reshape(2, 1))

        # Constant purely to visualise same as rate-based model
        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

    def cpu4_output(self, cpu4_mem):
        """Output activity based on memory."""
        return cpu4_mem

    def cpu1_output(self, tb1, cpu4):
        """Offset CPU4 columns by 1 column (45 degrees) left and right
        wrt TB1."""

        cpu4_reshaped = cpu4.reshape(2, -1)
        cpu1 = (1.0 - tb1) * np.vstack([np.roll(cpu4_reshaped[1], 1),
                                        np.roll(cpu4_reshaped[0], -1)])
        return cpu1.reshape(-1)

    def motor_output(self, cpu1, random_std=0.05):
        """Sum CPU1 to determine left or right turn."""
        cpu1_reshaped = cpu1.reshape(2, -1)
        motor_lr = np.sum(cpu1_reshaped, axis=1)
        # We need to add some randomness, otherwise agent infinitely overshoots
        motor = (motor_lr[1] - motor_lr[0])
        if random_std > 0.0:
            motor += np.random.normal(0, random_std)
        return motor

    def __str__(self):
        return "basic_holo"


class CXBasicForwards(CXBasic):
    """This class can't 'flip' the TB1 sinusoid, meaning it can integrate
    holonomically between -45 and 45 of forwards heading."""

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """Trying to be a bit more realistic, but only sensitive to motion
        in forward directions (-45 to +45 degrees)."""
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)
        # Signal comes in from PB. (inverse TB1 as inhibited.)
        # This is inhibited by TN1, the faster motion, the less inhibited
        mem_update = np.clip(0.5 - tn1.reshape(2, 1), 0, 1) * (1.0 - tb1)
        # Decay is proportionate to TN2
        mem_update -= 0.25 * tn2.reshape(2, 1)

        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

    def __str__(self):
        return "basic_pholo"


class CXBasicAveraging(CXBasicForwards):
    """Here CPU4 are averaged for each columns, to give OK path integration
    in most situations, however can get failure due to holonomic motion."""

    def tn1_output(self, flow):
        """Linearly inverse sensitive to forwards and backwards motion."""
        mean_flow = np.array([np.mean(flow), np.mean(flow)])
        return np.clip((1.0 - mean_flow) / 2.0, 0, 1)

    def tn2_output(self, flow):
        """Linearly sensitive to forwards motion only."""
        mean_flow = np.array([np.mean(flow), np.mean(flow)])
        return np.clip(mean_flow, 0, 1)

    def __str__(self):
        return "basic_av"


class CXBasicFlipped(CXBasic):
    """Here we are trying to invert TB1 preference angles to see if that
    results in a functioning path integrator."""

    def tb1_output(self, theta, tb1=None):
        """Sinusoidal response to solar compass."""
        return (1.0 + np.cos(np.pi + x - theta)) / 2.0

    def motor_output(self, cpu1, random_std=0.05):
        """Sum CPU1 to determine left or right turn."""
        cpu1_reshaped = cpu1.reshape(2, -1)
        motor_lr = np.sum(cpu1_reshaped, axis=1)
        # We need to add some randomness, otherwise agent infinitely overshoots
        motor = (motor_lr[1] - motor_lr[0])
        if random_std > 0.0:
            motor += np.random.normal(0, random_std)
        return motor

    def __str__(self):
        return "basic_holoflipped"
