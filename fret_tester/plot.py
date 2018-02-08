# Copyright 2017 Lukas Schrangl
"""Utilities for plotting results of the simulations"""
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class _PlotterBase:
    """Base class implementing common plotting functionality"""
    truth_style = ":"
    """Line style for plotting the true values"""

    truth_color = "C1"
    """Line color for plotting  the true values"""

    scale = "log"
    """Axes scale"""

    cbar_width = 0.03
    """Width of the colorbar as a fraction of 1.0"""

    tick_formatter = None
    """Tick formatter"""

    def __init__(self, **kwargs):
        """Parameters
        ----------
        **kwargs
            Set any of the class attributes (:py:attr:`truth_style`, …)
        """
        self.time_unit = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.tick_formatter is None:
            if self.scale == "log":
                self.tick_formatter = mpl.ticker.LogFormatter()
            else:
                self.tick_formatter = mpl.ticker.ScalarFormatter()

    @property
    def time_unit(self):
        """String describing the time unit

        This is used in axis labels.
        """
        return self._time_unit

    @time_unit.setter
    def time_unit(self, t):
        self._time_unit = t
        self._time_init_label = " [{}]".format(t) if t else ""

    def make_axes(self, nrows, n, ax_or_subspec, fig, colorbar="on"):
        """Create matplotlib Axes objects for plotting

        Parameters
        ----------
        nrows : int
            Number of rows. Only has an effect if `ax_or_subspec` is a
            :py:class:`SubplotSpec` instance.
        n : int
            Number plotss. Only has an effect if `ax_or_subspec` is a
            :py:class:`SubplotSpec` instance.
        ax_or_subspec : list of matplotlib Axes or matplotlib SupblotSpec or None
            If this is a :py:class:`SubplotSpec` instance, create new Axes on
            a grid in `ax_or_subspec`. If this is a list of Axes, just use
            those. If `None`, create a :py:class:`SubplotSpec` that fills the
            whole figure.
        fig : matplotlib.figure.Figure
            Figure to create the Axes on. Only applicable if `ax_or_subspec`
            is a :py:class:`SubplotSpec` instance.
        colorbar : {"on", "empty", "off"}, optional
            If "on" or "empty", create addtional Axes (of :py:attr:`cbar_width`
            width) intended for a colorbar (if `ax_or_subspec` is a
            :py:class:`SubplotSpec` instance) or use the last element of
            `ax_or_subspec` as colorbar Axes (if it is a list of Axes).
            Defaults to "on".

        Returns
        -------
        ax : list of matplotlib.axes.Axes
            Axes for plots
        cbar_ax : matplotlib.axes.Axes or None
            Axes for colorbar (if `colorbar` is "on" or "empty") or None
            (if `colorbar` is "off").
        """
        ncols = math.ceil(n / nrows)

        if ax_or_subspec is None:
            ax_or_subspec = mpl.gridspec.GridSpec(1, 1, hspace=0)[0]

        if colorbar in ("on", "empty"):
            col_width_ratios = (((1 - self.cbar_width) / ncols,) * ncols +
                                (self.cbar_width,))
            n_tot = n + nrows - 1
            if isinstance(ax_or_subspec, mpl.gridspec.SubplotSpec):
                grid = mpl.gridspec.GridSpecFromSubplotSpec(
                    nrows, ncols+1, ax_or_subspec,
                    width_ratios=col_width_ratios)
                ax = [fig.add_subplot(grid[i])
                      for i in range(n_tot) if i % (ncols + 1) != ncols]
                cbar_ax = (fig.add_subplot(grid[:, -1]) if colorbar == "on"
                           else None)
            else:
                if len(ax_or_subspec) != n + 1:
                    raise ValueError(
                        "`ax_or_subspec` should contain {} entries".format(
                            n + 1))
                ax = ax_or_subspec[:-1]
                cbar_ax = ax_or_subspec[-1]
            return ax, cbar_ax
        if colorbar == "off":
            if isinstance(ax_or_subspec, mpl.gridspec.GridSpecBase):
                grid = mpl.gridspec.GridSpecFromSubplotSpec(
                    nrows, ncols, ax_or_subspec)
                ax = [fig.add_subplot(grid[i]) for i in range(n)]
            else:
                if len(ax_or_subspec) != n:
                    raise ValueError(
                        "`ax_or_subspec` should contain {} entries".format(n))
                ax = ax_or_subspec
            return ax, None
        raise ValueError("colorbar must be in ('on', 'off', 'empty').")


class Plotter1D(_PlotterBase):
    """Class for producing 1D plots

    These are typically plots of lifetimes vs. p-values with fixed ratio
    between the two lifetimes.
    """
    pval_scale = "log"
    """Scale for the p-value axis"""

    significance = 1e-2
    """Significance threshold"""

    pval_range = (1e-3, 1.)
    """p-value axis range"""

    significance_opts = dict(color="gray", alpha=0.4, ls="None", lw=0.001)
    """Options for plotting the region of significance"""

    def __init__(self, **kwargs):
        """Parameters
        ----------
        **kwargs
            Set any of the class attributes (:py:attr:`truth_style`, …)
        """
        super().__init__(**kwargs)

        if not hasattr(self, "pval_tick_formatter"):
            if self.pval_scale == "log":
                self.pval_tick_formatter = mpl.ticker.LogFormatterSciNotation()
            else:
                self.pval_tick_formatter = mpl.ticker.ScalarFormatter()

    def plot(self, test_times, p_vals, truth=None, ax=None):
        """Draw a single 1D plot

        This is typically for plots of lifetimes vs. p-values with fixed ratio
        between the two lifetimes.

        Parameters
        ----------
        test_times : list of lists of float
            test_times[0] specifies the times the first lifetime :math:`\tau_1`
            was tested against. test_times[1] specify the times the second
            lifetime :math:`\tau_2` was tested against.
        p_vals : list of float
            p-values corresponding to `test_times`
        truth : tuple of float or None, optional
            If not `None`, draw lines at these time coordinates to
            mark the true lifetime combination (if known). Defaults to `None`.
        ax : matplotlib.axes.Axes or None:
            Axes to draw on. Use ``matplotlib.pyplot.gca()`` if `None`.
            Defaults to `None`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes for test_times[1]
        axt : matplotlib.axes.Axes
            Axes for test_times[0]
        """
        if ax is None:
            ax = plt.gca()

        if self.significance > min(self.pval_range):
            t = mpl.transforms.blended_transform_factory(ax.transAxes,
                                                         ax.transData)
            r = mpl.patches.Rectangle((0, min(self.pval_range)), 1,
                                      self.significance - min(self.pval_range),
                                      transform=t, **self.significance_opts)
            ax.add_patch(r)

        axt = ax.twiny()
        axt.plot(test_times[0], p_vals)

        axt.set_ylim(*self.pval_range)
        ax.set_xlim(np.min(test_times[1]), np.max(test_times[1]))
        axt.set_xlim(np.min(test_times[0]), np.max(test_times[0]))

        axt.set_xscale(self.scale)
        ax.set_xscale(self.scale)
        ax.set_yscale(self.pval_scale)

        if truth is not None:
            t = mpl.transforms.blended_transform_factory(axt.transData,
                                                         axt.transAxes)
            li = mpl.lines.Line2D([truth[0], truth[0]], [0, 1],
                                  transform=t, color=self.truth_color,
                                  linestyle=self.truth_style)
            axt.add_line(li)

        for a in (ax.xaxis, axt.xaxis):
            a.set_major_formatter(self.tick_formatter)
            a.set_minor_formatter(self.tick_formatter)
        ax.yaxis.set_major_formatter(self.pval_tick_formatter)
        ax.yaxis.set_minor_formatter(self.pval_tick_formatter)

        axt.set_xlabel("$\\tau_1$" + self._time_init_label)
        ax.set_xlabel("$\\tau_2$" + self._time_init_label)
        ax.set_ylabel("$p$-value")

        return ax, axt

    def plot_series(self, test_times, p_vals, truths=None, ax_or_subspec=None,
                    fig=None, nrows=1, empty_cbar=False):
        """Draw a series of 1D plots

        Use :py:meth:`plot` to draw a series of plots on a grid.

        Parameters
        ----------
        test_times : list
            Each entry will be passed to :py:meth:`plot` as `test_times`
            parameter.
        p_vals : list
            Each entry will be passed to :py:meth:`plot` as `p_vals` parameter.
        truths : list or None, optional
            Each entry will be passed to :py:meth:`plot` as `truth` parameter.
            Defaults to None.
        ax_or_subspec : list of matplotlib Axes or matplotlib SupblotSpec or None
            If this is a :py:class:`SubplotSpec` instance, create new Axes on
            a grid in `ax_or_subspec`. If this is a list of Axes, just use
            those. If `None`, create a :py:class:`SubplotSpec` that fills the
            whole figure.
        fig : matplotlib.figure.Figure or None
            Only effective if  `ax_or_subspec` is a :py:class:`SubplotSpec`
            instance. If `None`, use ``matplotlib.pyplot.gcf()``. Defaults to
            `None`.
        nrows : int, optional
            Number of rows. Defaults to 1.
        empty_cbar : bool, optional
            Whether to allocate an (emtpy) grid entry for a colorbar. This
            is useful if the plot series is paired with a 2D series (with
            a color bar) in the same figure. Defaults to False.

        Returns
        -------
        ax : list of matplotlib.axes.Axes
            Axes for ``tt[1]``, where ``tt`` stands for each entry of
            `test_times`.
        axt : list of matplotlib.axes.Axes
            Axes for ``tt[0]``, where ``tt`` stands for each entry of
            `test_times`.
        """
        n = len(test_times)
        ncols = math.ceil(n / nrows)

        if truths is None:
            truths = [None] * n
        empty_cbar = "empty" if empty_cbar else "off"

        if fig is None:
            fig = plt.gcf()

        ax, _ = self.make_axes(nrows, n, ax_or_subspec, fig, empty_cbar)

        axt = []
        for a, tt, p, tr in zip(ax, test_times, p_vals, truths):
            _, tw = self.plot(tt, p, tr, a)
            axt.append(tw)

        for i, a in enumerate(ax):
            if i % ncols == ncols - 1:
                a.tick_params("y", which="both", labelright=True, right=True,
                              labelleft=False, left=False)
                a.yaxis.set_label_position("right")
            else:
                a.tick_params("y", which="both", labelright=False, right=True,
                              labelleft=False, left=False)
                a.yaxis.label.set_visible(False)
        for a in axt[ncols:]:
            a.xaxis.label.set_visible(False)
            a.tick_params("x", which="both", labeltop=False, labelbottom=False)
        for a in ax[:ncols*(nrows-1)]:
            a.xaxis.label.set_visible(False)
            a.tick_params("x", which="both", labeltop=False, labelbottom=False)

        return ax, axt


class Plotter2D(_PlotterBase):
    """Class for producing 2D plots

    These are typically plots of where the two lifetimes were tested
    independently, producing a 2D plot with the p-values color-coded.
    """
    norm = mpl.colors.LogNorm()
    """p-value color map normalization. Logarithmic by default."""

    pval_range = (1e-2, 1.)
    """p-value range"""

    cmap = "binary"
    """p-value color map"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, test_times, p_vals, truth=None, ax=None):
        """Draw a single 2D plot

        These is typically for plots of where the two lifetimes were tested
        independently, producing a 2D plot with the p-values color-coded.

        Parameters
        ----------
        test_times : list of lists of float
            test_times[0] specifies the times the first lifetime :math:`\tau_1`
            was tested against. test_times[1] specify the times the second
            lifetime :math:`\tau_2` was tested against.
        p_vals : array_like, dtype(float), shape(len(test_times[0], len(test_times[1]))
            p-values corresponding to `test_times`
        truth : tuple of float or None, optional
            If not `None`, draw lines at these time coordinates to
            mark the true lifetime combination (if known). Defaults to `None`.
        ax : matplotlib.axes.Axes or None:
            Axes to draw on. Use ``matplotlib.pyplot.gca()`` if `None`.
            Defaults to `None`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes for test_times[1]
        axt : matplotlib.axes.Axes
            Axes for test_times[0]
        """
        if ax is None:
            ax = plt.gca()

        m = ax.pcolormesh(test_times[0], test_times[1], p_vals.T,
                          cmap=self.cmap, norm=self.norm,
                          vmin=self.pval_range[0], vmax=self.pval_range[1])
#        ax.contour(test_times[0], test_times[1], p_vals.T, [1e-6, 1e-3],
#                   linewidths=0.1, colors="k", linestyles=[":", "--"])

        if truth is not None:
            t1 = mpl.transforms.blended_transform_factory(ax.transData,
                                                          ax.transAxes)
            l1 = mpl.lines.Line2D([truth[0], truth[0]], [0, 1],
                                  transform=t1, color=self.truth_color,
                                  linestyle=self.truth_style)
            ax.add_line(l1)
            t2 = mpl.transforms.blended_transform_factory(ax.transAxes,
                                                          ax.transData)
            l2 = mpl.lines.Line2D([0, 1], [truth[1], truth[1]],
                                  transform=t2, color=self.truth_color,
                                  linestyle=self.truth_style)
            ax.add_line(l2)

        ax.set_xscale(self.scale)
        ax.set_yscale(self.scale)

        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(self.tick_formatter)
            a.set_minor_formatter(self.tick_formatter)

        ax.set_xlabel("$\\tau_1$" + self._time_init_label)
        ax.set_ylabel("$\\tau_2$" + self._time_init_label)

        return m

    def plot_series(self, test_times, p_vals, truths=None, ax_or_subspec=None,
                    fig=None, nrows=1):
        """Draw a series of 2D plots

        Use :py:meth:`plot` to draw a series of plots on a grid.

        Parameters
        ----------
        test_times : list
            Each entry will be passed to :py:meth:`plot` as `test_times`
            parameter.
        p_vals : list
            Each entry will be passed to :py:meth:`plot` as `p_vals` parameter.
        truths : list or None, optional
            Each entry will be passed to :py:meth:`plot` as `truth` parameter.
            Defaults to None.
        ax_or_subspec : list of matplotlib Axes or matplotlib SupblotSpec or None
            If this is a :py:class:`SubplotSpec` instance, create new Axes on
            a grid in `ax_or_subspec`. If this is a list of Axes, just use
            those. If `None`, create a :py:class:`SubplotSpec` that fills the
            whole figure.
        fig : matplotlib.figure.Figure or None
            Only effective if  `ax_or_subspec` is a :py:class:`SubplotSpec`
            instance. If `None`, use ``matplotlib.pyplot.gcf()``. Defaults to
            `None`.
        nrows : int, optional
            Number of rows. Defaults to 1.

        Returns
        -------
        ax : list of matplotlib.axes.Axes
            Axes for p-value plots.
        colorbar_ax : matplotlib.axes.Axes
            Axes for colorbar.
        fig : matplotlib.figure.Figure or None
            Only effective if  `ax_or_subspec` is a :py:class:`SubplotSpec`
            instance. If `None`, use ``matplotlib.pyplot.gcf()``. Defaults to
            `None`.
        nrows : int, optional
            Number of rows. Defaults to 1.

        Returns
        -------
        ax : list of matplotlib.axes.Axes
            Axes for p-value plots.
        colorbar_ax : matplotlib.axes.Axes
            Axes for colorbar.
        """
        n = len(test_times)
        ncols = math.ceil(n / nrows)

        if truths is None:
            truths = [None] * n

        if fig is None:
            fig = plt.gcf()

        ax, colorbar_ax = self.make_axes(nrows, n, ax_or_subspec, fig, "on")

        for a, tt, p, tr in zip(ax, test_times, p_vals, truths):
            m = self.plot(tt, p, tr, a)

        for a in (a2 for i, a2 in enumerate(ax) if i % ncols != 0):
            a.tick_params("y", which="both", labelright=False, labelleft=False)
            a.yaxis.label.set_visible(False)
        for a in ax[:ncols*(nrows-1)]:
            a.xaxis.label.set_visible(False)
            a.tick_params("x", which="both", labeltop=False, labelbottom=False)

        fig.colorbar(m, cax=colorbar_ax)
        colorbar_ax.set_ylabel(r"$p$-value")

        return ax, colorbar_ax
