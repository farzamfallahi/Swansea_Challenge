import numpy as np
import matplotlib.pyplot as plt

def plot_one_dimensional_histogram(data, bins=10, title='One-Dimensional Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Plots a one-dimensional histogram.
    
    :param data: List or array of data points.
    :param bins: Number of bins for the histogram.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.pause(0.001)  # Pause for a short time to allow the plot to be displayed

def plot_two_dimensional_histogram(x, y, bins=30, title='Two-Dimensional Histogram', xlabel='X Value', ylabel='Y Value'):
    """
    Plots a two-dimensional histogram (heatmap).
    
    :param x: List or array of x-axis data points.
    :param y: List or array of y-axis data points.
    :param bins: Number of bins for the histogram.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.hist2d(x, y, bins=bins, cmap='Blues')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.pause(0.001)  # Pause for a short time to allow the plot to be displayed

def plot_scatter(x, y, title='Scatter Plot', xlabel='X Value', ylabel='Y Value'):
    """
    Plots a scatter plot.
    
    :param x: List or array of x-axis data points.
    :param y: List or array of y-axis data points.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.pause(0.001)  # Pause for a short time to allow the plot to be displayed
