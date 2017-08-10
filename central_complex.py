import numpy as np

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B


def decode_position(cpu4_reshaped, cpu4_mem_gain):
    """Decode position from sinusoid in to polar coordinates.
    Amplitude is distance, Angle is angle from nest outwards.
    Without offset angle gives the home vector.
    Input must have shape of (2, -1)"""
    signal = np.sum(cpu4_reshaped, axis=0)
    fund_freq = np.fft.fft(signal)[1]
    angle = -np.angle(np.conj(fund_freq))
    distance = np.absolute(fund_freq) / cpu4_mem_gain
    return angle, distance


class CX(object):
    """Abstract base class for any central complex model."""

    def __init__(self, tn_prefs=np.pi/4.0,
                 cpu4_mem_gain=0.005):
        self.tn_prefs = tn_prefs
        self.cpu4_mem_gain = cpu4_mem_gain
        self.smoothed_flow = 0

    def tl2_output(self, theta):
        raise NotImplementedError("Subclasses should implement this!")

    def cl1_output(self, tl2):
        raise NotImplementedError("Subclasses should implement this!")

    def tb1_output(self, cl1, tb1):
        raise NotImplementedError("Subclasses should implement this!")

    def get_flow(self, heading, velocity, filter_steps=0):
        """Calculate optic flow depending on preference angles. [L, R]"""
        A = np.array([[np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)],
                      [np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)]])
        flow = np.dot(A, velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow

    def tn1_output(self, flow):
        raise NotImplementedError("Subclasses should implement this!")

    def tn2_output(self, flow):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu4_output(self, cpu4_mem):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu1_output(self, tb1, cpu4):
        raise NotImplementedError("Subclasses should implement this!")

    def motor_output(self, cpu1):
        """Positive output means turn left, negative means turn right."""
        raise NotImplementedError("Subclasses should implement this!")

    def decode_cpu4(self, cpu4):
        """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
        preference. When summed single sinusoid should point home."""
        cpu4_reshaped = cpu4.reshape(2, -1)
        cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                                  np.roll(cpu4_reshaped[1], -1)])
        return decode_position(cpu4_shifted, self.cpu4_mem_gain)
