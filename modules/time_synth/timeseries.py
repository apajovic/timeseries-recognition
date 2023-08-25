import matplotlib.pyplot as plt
import numpy as np
import random
import math


class TimeseriesBoundingbox:
    def __init__(self):
        self.position = 0
        self.width = 0

class Timeseries(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __init__(self, input_array):
        # Call the __init__ of the parent class if needed
        super().__init__()

        self.source_position = 0
        self.source_width = 0

    def set_source_position(self, value):
        self.source_position = value

    def set_source_width(self, value):
        self.source_width = value
    
    @property
    def length(self):
        return len(self)
    
    @property
    def time(self):
        return range(len(self))

    def plot_series(self, format="-", start=0, end=None, label=None,title=None, show=True):
        """
        Visualizes time series data

        Args:
        time (array of int) - contains the time steps
        series (array of int) - contains the measurements for each time step
        format (string) - line style when plotting the graph
        start (int) - first time step to plot
        end (int) - last time step to plot
        label (list of strings)- tag for the line
        """

        # Setup dimensions of the graph figure
        plt.figure(figsize=(10, 6))

        # Plot the time series data
        plt.plot(self.time[start:end], self[start:end], format)

        # Label the x-axis
        plt.xlabel("Time")

        # Label the y-axis
        plt.ylabel("Value")
        
        if title:
            plt.title(title, fontsize=14)
        if label:
            plt.legend(fontsize=14, labels=label)

        # Overlay a grid on the graph
        plt.grid(True)

        # Draw the graph on screen
        if show:
            # Draw the graph on screen
            plt.show()

        return self

    # Trend
    def add_trend(self, slope=0):
        """
        Generates synthetic data that follows a straight line given a slope value.

        Args:
        time (array of int) - contains the time steps
        slope (float) - determines the direction and steepness of the line

        Returns:
        series (array of float) - measurements that follow a straight line
        """

        # Compute the linear series given the slope
        return Timeseries(self +  slope * np.indices(self.shape))

    # Noise
    def add_noise(self, noise_level=1, seed=None):
        """Generates a normally distributed noisy signal

        Args:
        time (array of int) - contains the time steps
        noise_level (float) - scaling factor for the generated signal
        seed (int) - number generator seed for repeatability

        Returns:
        noise (array of float) - the noisy signal

        """

        # Initialize the random number generator
        rnd = np.random.RandomState(seed)

        # Generate a random number for each time step and scale by the noise level
        noise = rnd.randn(len(self)) * noise_level # TODO maybe try with normal or uniform distribution

        return Timeseries(self + noise)

    def increase_precision_quadratic(self, precision_level):
        array = self.copy()
        for i in range(precision_level):
            array_x = array.repeat(2)
            array = (array_x[1:]+array_x[:-1]) / 2
        return Timeseries(array)
    
    
    def increase_precision(self, precision_level):
        def create_line(a, b, num_dots):
            return np.array([a + i * (b - a)/num_dots for i in range(num_dots)])
        
        array = self.copy()
        return  Timeseries(np.concatenate([
            create_line(array[i], array[i+1], precision_level) for i in range(len(array)-1)
            ]))
        
    def randomly_increase_precision(self, seed=(3, 7)):
        array = self.copy()
        return Timeseries(np.concatenate([
            self[i:i+2].increase_precision(random.randint(*seed))
                for i in range(len(array) - 1)]))

    def downsample_to(self, num_samples):
        if self.length < num_samples:
            return self

        step = self.length // num_samples + 1
        return self[::step]

    def add_random_padding(self, padding_to_signal_ratio: int = 0, num_samples = 0, noise_level: float = 0.1, side='left', offset=0.5):
        def noisy_pad(vec, pad_width, *_, **__):
            vec[:pad_width[0]] = offset + np.random.randn(pad_width[0]) * noise_level
            vec[vec.size-pad_width[1]:] = offset + np.random.randn(pad_width[1]) * noise_level

        pad_size = int(len(self) * padding_to_signal_ratio) if num_samples == 0 else max(int(num_samples) - len(self), 0)
        if side == 'left':
            right = 0
            left = pad_size // random.randint(1, 10) if num_samples == 0 else int(pad_size)
        elif side == 'right':
            left = 0
            right = pad_size // random.randint(1, 10) if num_samples == 0 else int(pad_size)
        else:
            ratio = random.randint(1,10)
            left = math.floor(pad_size / ratio)
            right = pad_size - left 
        
        return Timeseries(np.pad(self, (left, right), mode=noisy_pad))


    def create_padding(self, point_noise_level=0.4, noise_level=0.15, max_size=200):
        # Think about this, this whole thing is a repeat of the pattern signal
        padding_signal = [self[-1]] + [0.5] * 10
        return Timeseries(padding_signal)\
                            .add_noise(point_noise_level)\
                            .increase_precision_quadratic(4)\
                            .add_noise(noise_level)\
                            .downsample_to(max_size)
        

    def pad_signal(self, point_noise_level=0.4, noise_level=0.15, max_size=200, side='left', source_bb:TimeseriesBoundingbox = TimeseriesBoundingbox()):
        final_signal = Timeseries(np.zeros(max_size))

        padding_signal = self.create_padding(point_noise_level, noise_level, max_size)
        while(len(padding_signal) + len(self) < max_size):
            # Add more padding in case the first one wasn't enough
            padding_signal = np.concatenate([padding_signal, self.create_padding(point_noise_level, noise_level, max_size)])

        # Take only a part of it to get up to max_size
        padding_signal = padding_signal[:max_size-len(self)] 
        pad_size = len(padding_signal)

        if side == 'left':
            final_signal[:len(padding_signal)] = padding_signal  
            final_signal[len(padding_signal):] = self
            source_bb.position = len(padding_signal)
        elif side == 'right':
            final_signal[:len(self)] = self  
            final_signal[len(self):] = padding_signal
            source_bb.position = 0
        else:
            ratio = random.randint(1,10)
            left = math.floor(pad_size / ratio)
            final_signal[:left] = padding_signal[:left]
            final_signal[left:left + len(self)] = self
            final_signal[left + len(self):] = padding_signal[left:]
            source_bb.position = left

        source_bb.width = len(self)

        return final_signal

    def normalize(self):
        return (self - np.min(self)) / (np.max(self) - np.min(self))

    def offset(self, offset_limit, seed=None):
        rnd = np.random.RandomState(seed)

        # Generate a random number for each time step and scale by the noise level
        rnd_offset = rnd.standard_normal() * offset_limit
        offset_signal = np.ones(len(self)) * rnd_offset

        return Timeseries(self + offset_signal)

    def random_choice(self, min_length, max_length):
        if min_length > len(self):
            # Really no point in doing this
            return self
        n = random.randint(min_length, min(len(self), max_length))
        index = np.full(self.shape, False, bool)
        index[np.random.choice(np.arange(index.shape[0]), n, replace = False)] = True

        return self[index]

    
