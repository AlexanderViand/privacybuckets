#!/usr/bin/python
# coding=utf-8
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def delta_eps_plot(plot_name, filename):
    data = np.genfromtxt(filename, delimiter=',')

    fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0)  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    # fig_height = (fig_width * 1)  # height in inches
    fig_size = [0.95 * fig_width, 0.95 *fig_height]

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'font.size': 18,
              'font.family': 'times new roman'}

    pdf_pages = PdfPages(os.path.join(plot_name))

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors = ['0.1', '0.2', '0.3', '0.4']
    linestyles = ['-', ':', '--', '-.']
    dotstyle = ['', '', '', '']


    plt.plot(data[:, 0], data[:, 1], dotstyle[0], color=colors[0], linestyle=linestyles[0], linewidth=2, zorder=3,
             label=r'$\mathcal{{N}}(0,{{{:d}}}^2)$'.format(150))
    plt.plot(data[:, 0], data[:, 2], dotstyle[0], color=colors[1], linestyle=linestyles[1], linewidth=2, zorder=3,
             label=r'$2 \cdot \mathcal{{B}}({{{:d}}}, \frac{{1}}{{2}})$'.format(22500))
    plt.plot(data[:, 0], data[:, 3], dotstyle[0], color=colors[2], linestyle=linestyles[2], linewidth=2, zorder=3,
             label=r'$3 \cdot \mathcal{{B}}({{{:d}}}, \frac{{1}}{{2}})$'.format(10000))
    plt.plot(data[:, 0], data[:, 4], dotstyle[0], color=colors[3], linestyle=linestyles[3], linewidth=2, zorder=3,
             label=r'$4 \cdot \mathcal{{B}}({{{:d}}}, \frac{{1}}{{2}})$'.format(5625))

    delta_point = 0.5 * 10 ** -5
    pointG = np.abs(np.array(data[:, 1]) - delta_point).argmin()
    plt.text(data[pointG, 0], 1.1 * delta_point, "$\epsilon={{{:.2f}}}$".format(data[pointG, 0]))
    pointB4 = np.abs(np.array(data[:, 4]) - delta_point).argmin()
    plt.text(data[pointB4, 0], 1.1 * delta_point, "$\epsilon ={{{:.2f}}}$".format(data[pointB4, 0]))

    plt.legend()
    plt.xlabel('$\epsilon$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.ylabel('$\delta$')
    plt.ylim(ymin=0, ymax=1.5 * 10 ** -5)
    plt.xlim(xmin=0.5, xmax=2.5)
    plt.axhline(y=delta_point, color='0.4', linestyle='--')



    #plt.legend(bbox_to_anchor=(-0.01, 0.75, 1., .102), loc=3, ncol=2, columnspacing=0.6, borderpad=0.3)

    plt.grid(True, linestyle=':', color='0.6', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight', pad_inches=0.1)
    print "generated " + plot_name
    plt.clf()
    pdf_pages.close()




delta_eps_plot("delta-eps.pdf", "delta-eps.csv")
