from central_complex import *
import numpy as np
from scipy.special import expit


def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def noisify_weights(W, noise=0.01):
    """Takes a weight matrix and adds some noise on to non-zero values."""
    N = np.random.normal(scale=noise, size=W.shape)
    # Only noisify the connections (positive values in W). Not the zeros.
    N_nonzero = N * W
    return W + N_nonzero


# TUNED PARAMETERS:
tl2_slope_tuned = 6.8
tl2_bias_tuned = 3.0

cl1_slope_tuned = 3.0
cl1_bias_tuned = -0.5

tb1_slope_tuned = 5.0
tb1_bias_tuned = 0.0

cpu4_slope_tuned = 5.0
cpu4_bias_tuned = 2.5

cpu1_slope_tuned = 5.0
cpu1_bias_tuned = 2.5

motor_slope_tuned = 1.0
motor_bias_tuned = 3.0


class CXRate(CX):
    """Class to keep a set of parameters for a model together.
    No state is held in the class currently."""
    def __init__(self,
                 noise=0.1,
                 tl2_slope=tl2_slope_tuned,
                 tl2_bias=tl2_bias_tuned,
                 tl2_prefs=np.tile(np.linspace(0, 2*np.pi, N_TB1,
                                               endpoint=False), 2),
                 cl1_slope=cl1_slope_tuned,
                 cl1_bias=cl1_bias_tuned,
                 tb1_slope=tb1_slope_tuned,
                 tb1_bias=tb1_bias_tuned,
                 cpu4_slope=cpu4_slope_tuned,
                 cpu4_bias=cpu4_bias_tuned,
                 cpu1_slope=cpu1_slope_tuned,
                 cpu1_bias=cpu1_bias_tuned,
                 motor_slope=motor_slope_tuned,
                 motor_bias=motor_bias_tuned,
                 weight_noise=0.0,
                 **kwargs):

        super(CXRate, self).__init__(**kwargs)
        # Default noise used by the model for all layers
        self.noise = noise

        # Weight matrices based on anatomy. These are not changeable!)
        self.W_CL1_TB1 = np.tile(np.eye(N_TB1), 2)
        self.W_TB1_TB1 = gen_tb_tb_weights()
        self.W_TB1_CPU1a = np.tile(np.eye(N_TB1), (2, 1))[1:N_CPU1A+1, :]
        self.W_TB1_CPU1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0]])
        self.W_TB1_CPU4 = np.tile(np.eye(N_TB1), (2, 1))
        self.W_TN_CPU4 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            ]).T
        self.W_CPU4_CPU1a = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ])
        self.W_CPU4_CPU1b = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
            ])
        self.W_CPU1a_motor = np.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.W_CPU1b_motor = np.array([[0, 1],
                                       [1, 0]])

        if weight_noise > 0.0:
            self.W_CL1_TB1 = noisify_weights(self.W_CL1_TB1, weight_noise)
            self.W_TB1_TB1 = noisify_weights(self.W_TB1_TB1, weight_noise)
            self.W_TB1_CPU1a = noisify_weights(self.W_TB1_CPU1a, weight_noise)
            self.W_TB1_CPU1b = noisify_weights(self.W_TB1_CPU1b, weight_noise)
            self.W_TB1_CPU4 = noisify_weights(self.W_TB1_CPU4, weight_noise)
            self.W_CPU4_CPU1a = noisify_weights(self.W_CPU4_CPU1a,
                                                weight_noise)
            self.W_CPU4_CPU1b = noisify_weights(self.W_CPU4_CPU1b,
                                                weight_noise)
            self.W_CPU1a_motor = noisify_weights(self.W_CPU1a_motor,
                                                 weight_noise)
            self.W_CPU1b_motor = noisify_weights(self.W_CPU1b_motor,
                                                 weight_noise)
        # The cell properties (for sigmoid function)
        self.tl2_slope = tl2_slope
        self.tl2_bias = tl2_bias
        self.tl2_prefs = tl2_prefs
        self.cl1_bias = cl1_bias
        self.cl1_slope = cl1_slope
        self.tb1_slope = tb1_slope
        self.tb1_bias = tb1_bias
        self.cpu4_slope = cpu4_slope
        self.cpu4_bias = cpu4_bias
        self.cpu1_slope = cpu1_slope
        self.cpu1_bias = cpu1_bias
        self.motor_slope = motor_slope
        self.motor_bias = motor_bias

    def tl2_output(self, theta):
        """Just a dot product with preferred angle and current heading"""
        output = np.cos(theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def cl1_output(self, tl2):
        """Takes input from the TL2 neurons and gives output."""
        return noisy_sigmoid(-tl2, self.cl1_slope, self.cl1_bias, self.noise)

    def tb1_output(self, cl1, tb1):
        """Ring attractor state on the protocerebral bridge."""
        prop_cl1 = 0.667   # Proportion of input from CL1 vs TB1
        prop_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(self.W_CL1_TB1, cl1) -
                  prop_tb1 * np.dot(self.W_TB1_TB1, tb1))
        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def tn1_output(self, flow):
        output = (1.0 - flow) / 2.0
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def tn2_output(self, flow):
        output = flow
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """Memory neurons update.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        cpu4_mem += (np.clip(np.dot(self.W_TN_CPU4, 0.5-tn1), 0, 1) *
                     self.cpu4_mem_gain * np.dot(self.W_TB1_CPU4, 1.0-tb1))

        cpu4_mem -= self.cpu4_mem_gain * 0.25 * np.dot(self.W_TN_CPU4, tn2)
        return np.clip(cpu4_mem, 0.0, 1.0)

    def cpu4_output(self, cpu4_mem):
        """The output from memory neuron, based on current calcium levels."""
        return noisy_sigmoid(cpu4_mem, self.cpu4_slope,
                             self.cpu4_bias, self.noise)

    def cpu1a_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = np.dot(self.W_CPU4_CPU1a, cpu4) * np.dot(self.W_TB1_CPU1a,
                                                          1.0-tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1b_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = np.dot(self.W_CPU4_CPU1b, cpu4) * np.dot(self.W_TB1_CPU1b,
                                                          1.0-tb1)

        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1_output(self, tb1, cpu4):
        cpu1a = self.cpu1a_output(tb1, cpu4)
        cpu1b = self.cpu1b_output(tb1, cpu4)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    def motor_output(self, cpu1):
        """outputs a scalar where sign determines left or right turn."""
        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = np.dot(self.W_CPU1a_motor, cpu1a)
        motor += np.dot(self.W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1]) * 0.25  # To kill the noise a bit!
        return output

    def __str__(self):
        return "rate_pholo"


class CXRatePontin(CXRate):

    def __init__(self, *args, **kwargs):

        super(CXRatePontin, self).__init__(**kwargs)
        self.cpu4_mem_gain *= 0.5
        self.cpu1_bias = -1.0
        self.cpu1_slope = 7.5

        # Pontine cells
        self.pontin_slope = 5.0
        self.pontin_bias = 2.5

        self.W_pontin_CPU1a = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15
            ])
        self.W_pontin_CPU1b = np.array([
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #9
            ])
        self.W_CPU4_pontin = np.eye(N_CPU4)

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """Memory neurons update.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        mem_update = np.dot(self.W_TN_CPU4, tn2)
        mem_update -= np.dot(self.W_TB1_CPU4, tb1)
        mem_update = np.clip(mem_update, 0, 1)
        mem_update *= self.cpu4_mem_gain
        cpu4_mem += mem_update
        cpu4_mem -= 0.125 * self.cpu4_mem_gain

        return np.clip(cpu4_mem, 0.0, 1.0)

    def pontin_output(self, cpu4):
        inputs = np.dot(self.W_CPU4_pontin, cpu4)
        return noisy_sigmoid(inputs, self.pontin_slope, self.pontin_bias,
                             self.noise)

    def cpu1a_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = 0.5 * np.dot(self.W_CPU4_CPU1a, cpu4)

        pontin = 0.5 * self.pontin_output(cpu4)
        inputs -= np.dot(self.W_pontin_CPU1a, pontin)
        inputs -= np.dot(self.W_TB1_CPU1a, tb1)

        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1b_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = 0.5 * np.dot(self.W_CPU4_CPU1b, cpu4)

        pontin = 0.5 * self.pontin_output(cpu4)
        inputs -= np.dot(self.W_pontin_CPU1b, pontin)
        inputs -= np.dot(self.W_TB1_CPU1b, tb1)

        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def decode_cpu4(self, cpu4):
        """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
        preference. When summed single sinusoid should point home."""
        cpu4_reshaped = cpu4.reshape(2, -1)
        cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                                  np.roll(cpu4_reshaped[1], -1)])
        return decode_position(cpu4_shifted, self.cpu4_mem_gain*2.0)

    def __str__(self):
        return "rate_pontin"


class CXRateAveraging(CXRate):
    def tn1_output(self, flow):
        mean_flow = np.array([np.mean(flow), np.mean(flow)])
        output = (1.0 - mean_flow) / 2.0
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def tn2_output(self, flow):
        output = np.array([np.mean(flow), np.mean(flow)])
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def __str__(self):
        return "rate_av"


class CXRateHolonomic(CXRate):
    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        # TODO(tomish) Fix this to make it more realistic
        cpu4_mem += (np.dot(self.W_TN_CPU4, 0.5-tn1) * self.cpu4_mem_gain *
                     np.dot(self.W_TB1_CPU4, 1.0-tb1))

        cpu4_mem -= self.cpu4_mem_gain * 0.25 * np.dot(self.W_TN_CPU4, tn2)

        return np.clip(cpu4_mem, 0.0, 1.0)

    def __str__(self):
        return "rate_holo"


class CXRatePontinAveraging(CXRatePontin):
    def tn1_output(self, flow):
        mean_flow = np.array([np.mean(flow), np.mean(flow)])
        output = (1.0 - mean_flow) / 2.0
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def tn2_output(self, flow):
        output = np.array([np.mean(flow), np.mean(flow)])
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def __str__(self):
        return "rate_pontin_av"


class CXRatePontinHolonomic(CXRatePontin):
    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)
        mem_update = (0.5 - tn1.reshape(2, 1)) * (1.0 - tb1)
        mem_update -= 0.5 * (0.5 - tn1.reshape(2, 1))

        # Constant purely to visualise same as rate-based model
        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

        return cpu4_mem

    def decode_cpu4(self, cpu4):
        """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
        preference. When summed single sinusoid should point home."""
        cpu4_reshaped = cpu4.reshape(2, -1)
        cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                                  np.roll(cpu4_reshaped[1], -1)])
        return decode_position(cpu4_shifted, self.cpu4_mem_gain)

    def __str__(self):
        return "rate_pontin_holo"