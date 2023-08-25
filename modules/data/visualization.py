import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.patches import Rectangle
from modules.time_synth.tools import plot_series
from modules.data.loader import load_dataset, decode_one_hot
from tensorflow import keras


def visualize_prediction(
        time_series,
        predicted_class: str, actual_class: str,
        predicted_range: tuple[float, float], actual_range: tuple[float, float]):
    title = f'Predicted: {predicted_class.capitalize()} / Actual: {actual_class.capitalize()}'
    time_series = tf.reshape(time_series, (len(time_series), 1))
    plot_series(range(len(time_series)), time_series, title=title, show=False)

    actual_rect = Rectangle((actual_range[0], 0), actual_range[1], 1, color='orange')
    actual_rect.set_alpha(0.3)
    predicted_rect = Rectangle((predicted_range[0], 0), predicted_range[1], 1, color='green')
    predicted_rect.set_alpha(0.3)
    plt.gca().add_patch(actual_rect)
    plt.gca().add_patch(predicted_rect)

    plt.show()


def grad_cam(model: keras.Model, data):
    """
    This function calculates the parameter activation of the model caused by the provided time series
    This is useful when you want to see which parts of the time series the model used to give its prediction
    Params:
        model (keras.Model) : the trained model that will evaluate the time series
        data (tf.Tensor) : the time series that will be evaluated
    Returns:
        predictions (tf.Tensor) : probabilities of each of the classes
        heatmap (tf.Tensor) : the activations of each of the parameters for the predicted class
    """
    predictions, last_conv_layer_output = model(data)  # initialize the model

    with tf.GradientTape() as tape:
        predictions, last_conv_layer_output = model(data)
        prediction_index = tf.argmax(predictions, axis=1)
        class_channel = predictions[:, int(prediction_index)]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=0)

    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=1)
    heatmap = tf.expand_dims(heatmap, axis=0)

    return predictions, heatmap


def plot_heatmap(time_series, heatmap, title='', show=True, debug=False):
    tf.expand_dims(heatmap, axis=2)

    if debug:
        plt.figure(figsize=(10, 6))

    plt.imshow(
        heatmap,
        cmap='PuOr',
        aspect="auto", interpolation='nearest',
        extent=[0, time_series.shape[1], tf.reduce_min(time_series), tf.reduce_max(time_series)],
        alpha=0.5
    )
    plt.plot(range(time_series.shape[1]), tf.squeeze(time_series), 'k')
    plt.colorbar()
    plt.title(title)
    
    if show:
        plt.show()


def predict_and_plot(
        model: tf.keras.Model,
        time_series: tf.Tensor,
        title = '',
        show=True, debug=False):
    """
    This function receives a model and a time series for which it predicts its class.
    After that, it draws the time series and GradCAM activation in the model parameters for that time series.
    Params:
        model (keras.Model) : the model that will be used for prediction
        time_series (Tensor) : the time series that will be evaluated by the model
        classes (list[str]) : the list of class names the model can recognize
        actual (Tensor) : one hot encoding of the actual class
        show : whether the graph will be showed, used if you want to stack multiple of these graphs in a larger one
        debug : whether the figure will be created, used if you want to stack multiple of these graphs in a larger one
    """

    if len(time_series.shape) == 2:
        time_series = tf.expand_dims(time_series, axis=0)

    prediction, heatmap = grad_cam(model, time_series)
    plot_heatmap(time_series, heatmap, title, show, debug)


if __name__ == '__main__':
    data, _ = load_dataset()
    for i, element in enumerate(data.as_numpy_iterator()):
        ts = element[0]
        predicted_class = 'pred'
        actual_class = 'actual'
        predicted_range = (10, 20)
        actual_range = (15, 25)
        visualize_prediction(ts, predicted_class, actual_class, predicted_range, actual_range)
        break
