"""Plotting utility functions."""

import matplotlib.pyplot as plt

from .config import trilobite_config


def resolve_fig_axes(fig=None, axes=None, fig_size=None):
    """
    Resolve and return a figure and axes for plotting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        An existing figure object. If None, a new figure will be created.
    axes : matplotlib.axes.Axes, optional
        An existing axes object. If None, axes will be created or retrieved from the figure.
    fig_size : tuple, optional
        Size of the figure to create if fig is None. Default is (8, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resolved figure object.
    """
    import matplotlib.pyplot as plt

    if fig_size is None:
        fig_size = trilobite_config["plotting.default_figsize"]

    if fig is None and axes is None:
        fig, axes = plt.subplots(figsize=fig_size)
    elif fig is not None and axes is None:
        axes = fig.gca()
    elif fig is None and axes is not None:
        fig = axes.figure

    return fig, axes


def set_plot_style():
    """Set the global plot style for matplotlib figures."""
    plt.rcParams["text.usetex"] = trilobite_config["plotting.use_tex"]

    if trilobite_config["plotting.use_tex"]:
        plt.rcParams["text.latex.preamble"] = trilobite_config["plotting.latex_preamble"]

    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.major.size"] = 8
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def get_default_cmap():
    """Return the default colormap.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The default colormap.
    """
    return plt.get_cmap(trilobite_config["plotting.default_cmap"])


def get_cmap(cmap):
    """
    Return a Matplotlib colormap from either a colormap instance or a name.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Either a Matplotlib colormap instance or the name of a registered
        Matplotlib colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        A Matplotlib colormap object.

    Raises
    ------
    TypeError
        If ``cmap`` is neither a string nor a Colormap instance.
    ValueError
        If ``cmap`` is a string but does not correspond to a known colormap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap

    if isinstance(cmap, Colormap):
        return cmap

    if isinstance(cmap, str):
        try:
            return plt.get_cmap(cmap)
        except ValueError as exc:
            raise ValueError(f"Unknown colormap name '{cmap}'.") from exc

    raise TypeError("cmap must be either a matplotlib.colors.Colormap instance or a string colormap name.")
