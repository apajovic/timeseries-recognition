import matplotlib.pyplot as plt
import numpy as np
import random
import math

def plot_series(time, series, format="-", start=0, end=None, title=None, label=None, xlabel='Time', ylabel='Value', fontsize=7,  show=True):
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


    # Plot the time series data
    plt.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    if title:
      plt.title(title, fontsize=fontsize)

    if label:
      plt.legend(fontsize=fontsize, labels=label)

    # Overlay a grid on the graph
    plt.grid(True)

    if show:
      # Draw the graph on screen
      plt.show()


def plot_series_on_axis(ax, time, series, format="-", start=0, end=None, title=None, label=None, show=True):
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

    # Plot the time series data
    ax.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    ax.xlabel("Time")

    # Label the y-axis
    ax.ylabel("Value")

    if title:
      ax.title(title, fontsize=14)

    if label:
      ax.legend(fontsize=14, labels=label)

    # Overlay a grid on the graph
    ax.grid(True)
    
    return ax

# Trend
def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value.

    Args:
      time (array of int) - contains the time steps
      slope (float) - determines the direction and steepness of the line

    Returns:
      series (array of float) - measurements that follow a straight line
    """

    # Compute the linear series given the slope
    series = slope * time

    return series

# Seasonality
def seasonal_pattern(season_time):
    """
    Just an arbitrary pattern, you can change it if you wish

    Args:
      season_time (array of float) - contains the measurements per time step

    Returns:
      data_pattern (array of float) -  contains revised measurement values according
                                  to the defined pattern
    """

    # Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    """
    Repeats the same pattern at each period

    Args:
      time (array of int) - contains the time steps
      period (int) - number of time steps before the pattern repeats
      amplitude (int) - peak measured value in a period
      phase (int) - number of time steps to shift the measured values

    Returns:
      data_pattern (array of float) - seasonal data scaled by the defined amplitude
    """

    # Define the measured values per period
    season_time = ((time + phase) % period) / period

    # Generates the seasonal data scaled by the defined amplitude
    data_pattern = amplitude * seasonal_pattern(season_time)

    return data_pattern

# Noise
def noise(time, noise_level=1, seed=None):
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
    noise = rnd.randn(len(time)) * noise_level

    return noise

# Impulsivity
def impulses(time, num_impulses, amplitude=1, seed=None):
    """
    Generates random impulses

    Args:
      time (array of int) - contains the time steps
      num_impulses (int) - number of impulses to generate
      amplitude (float) - scaling factor
      seed (int) - number generator seed for repeatability

    Returns:
      series (array of float) - array containing the impulses
    """

    # Initialize random number generator
    rnd = np.random.RandomState(seed)

    # Generate random numbers
    impulse_indices = rnd.randint(len(time), size=num_impulses)

    # Initialize series
    series = np.zeros(len(time))

    # Insert random impulses
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude

    return series

def autocorrelation_impulses(source, phis):
    """
    Generates autocorrelated data from impulses

    Args:
      source (array of float) - contains the time steps with impulses
      phis (dict) - dictionary containing the lag time and decay rates

    Returns:
      ar (array of float) - generated autocorrelated data
    """

    # Copy the source
    ar = source.copy()

    # Compute new series values based on the lag times and decay rates
    for step, value in enumerate(source):
        for lag, phi in phis.items():
            if step - lag > 0:
              ar[step] += phi * ar[step - lag]

    return ar

def increase_precision_quadratic(source, precision_level):
    array = source
    for i in range(precision_level):
        array_x = array.repeat(2)
        array = (array_x[1:]+array_x[:-1]) / 2
    return array

def increase_precision(source, precision_level):
    def create_line(a, b, num_dots):
        return np.array([a + i * (b - a)/num_dots for i in range(num_dots)])
    
    array = source
    return np.concatenate([
        create_line(array[i], array[i+1], precision_level) for i in range(len(array)-1)
        ])

def randomly_increase_precision(source, random_number_range=(1,10)):
    return np.concatenate([
        increase_precision(source[i:i+2], random.randint(*random_number_range))
          for i in range(len(source) - 1)])

def downsample_to(source, num_samples):
    if len(source) < num_samples:
        return source

    step = len(source) // num_samples + 1
    return source[::step]

def add_random_padding(source: np.ndarray, padding_to_signal_ratio: int = 0, num_samples = 0, noise_level: float = 0.1, side='left', offset=0.5):
    def noisy_pad(vec, pad_width, *_, **__):
        vec[:pad_width[0]] = offset + noise(range(pad_width[0]), noise_level)
        vec[vec.size-pad_width[1]:] = offset + noise(range(pad_width[1]), noise_level)

    pad_size = int(len(source) * padding_to_signal_ratio) if num_samples == 0 else max(int(num_samples) - len(source), 0)
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
        
    return np.pad(source, (left, right), mode=noisy_pad)

def create_padding(point_noise_level=0.4, noise_level=0.15, max_size=200):
    # Think about this, this whole thing is a repeat of the pattern signal
    padding_signal = [0.5] * 10
    padding_signal = np.array(padding_signal)
    padding_signal += noise(range(len(padding_signal)), point_noise_level)
    padding_signal = increase_precision_quadratic(padding_signal, 4)
    padding_signal += noise(range(len(padding_signal)), noise_level)
    padding_signal = downsample_to(padding_signal, max_size)
    return padding_signal

def pad_signal(source, point_noise_level=0.4, noise_level=0.15, max_size=200, side='left'):
    final_signal = np.zeros(max_size)

    padding_signal = create_padding(point_noise_level, noise_level, max_size)
    while(len(padding_signal) + len(source) < max_size):
        # Add more padding in case the first one wasn't enough
        padding_signal = np.concatenate([padding_signal, create_padding(point_noise_level, noise_level, max_size)])

    # Take only a part of it to get up to max_size
    padding_signal = padding_signal[:max_size-len(source)] 
    pad_size = len(padding_signal)

    if side == 'left':
        final_signal[:len(padding_signal)] = padding_signal  
        final_signal[len(padding_signal):] = source
        source_position = len(padding_signal)
    elif side == 'right':
        final_signal[:len(source)] = source  
        final_signal[len(source):] = padding_signal
        source_position = 0
    else:
        ratio = random.randint(1,10)
        left = math.floor(pad_size / ratio)
        final_signal[:left] = padding_signal[:left]
        final_signal[left:left + len(source)] = source
        final_signal[left + len(source):] = padding_signal[left:]
        source_position = left


    return final_signal, source_position, len(source)

def normalize(source):
    return (source - np.min(source)) / (np.max(source) - np.min(source))

