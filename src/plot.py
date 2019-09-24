import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
import matplotlib.pyplot as plt
import tkinter as Tk


def set_plot_param(x_values, y_values):
    parameters = [max(x_values) + min(x_values),  # plot x_range
                  max(y_values) + min(y_values)]  # plot y_range
    return parameters


def plot_spectrum(x_values, y_values):
    plt.clf()
    plt.cla()
    # Get & set plot parameters
    range_parameters = set_plot_param(x_values, y_values)
    axes = plt.gca()
    axes.set_xlim([0, range_parameters[0]])
    axes.set_ylim([0, range_parameters[1]])
    axes.grid()

    # Set plot attributes
    plt.subplots_adjust(left=0.12)
    plt.figure(figsize=(5, 2.5))
    plt.plot(x_values, y_values, color='blue')
    plt.style.use('seaborn')
    plt.grid(True)

    # Set plot labels
    plt.xlabel('m/z ratio (kg/C)', labelpad=None)
    plt.ylabel('intensity (c/s)', labelpad=None)

    # Set plot to a figure
    figure = plt.gcf()
    figure.tight_layout(pad=0.9)
    return figure


def draw_figure(canvas, figure, loc=(0, 0)):
    # Convert plot figure into a photo object
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
    canvas.image = photo
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    plt.close('all')
    return photo
