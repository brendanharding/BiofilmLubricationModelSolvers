"""
Convenience functions for plotting the output of 
the one dimensional biofilm lubrication model class.

Maintainer: Brendan Harding
Last updated: August 2020
"""

import matplotlib.pyplot as plt

def Plot1DFields(r,h,phi_n_bar,g_s,g_b):
    """
    Generates a nice plot of the 1D fields with 2 axes and a legend.
    Note: The sizing works well in a jupyter notebook 
    but probably should be adjusted for a paper.
    """
    fig,ax1 = plt.subplots(figsize=(6.7,4))
    fig.subplots_adjust(right=0.8)
    ax2 = ax1.twinx()
    p1, = ax1.plot(r,h,'C0-',label=r'$h$')
    p2, = ax2.plot(r,phi_n_bar,'C1-',label=r'$\bar{\phi}_n$')
    p3, = ax2.plot(r,g_s,'C2-',label=r'$g_s$')
    p4, = ax2.plot(r,g_b,'C3-',label=r'$g_b$')
    ax1.set_xlabel(r'$r$',labelpad=0)
    ax1.set_ylabel(r'$h$',rotation=0,labelpad=10)
    ax1.set_xlim(r[0],r[-1])
    ax2.set_ylabel('$\\bar{\\phi}_n$\n$g_s$\n$g_b$',rotation=0,labelpad=12,va='center')
    ax2.set_ylim(-0.05,1.05)
    lines = [p1,p2,p3,p4]
    ax1.legend(lines,[l.get_label() for l in lines],loc='center left',bbox_to_anchor=(1.16,0.54))
    return fig,[ax1,ax2]
