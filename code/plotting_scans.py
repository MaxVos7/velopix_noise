### Import Modules ###
import sys
import math
import glob
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm


### Plot Scan Summary ###
def plot_scans(filename, dacMin, dacMax):
    velopix = filename.split('_')[-1]
    files = sorted(glob.glob(filename + "_Trim*_Noise_Mean.csv"))

    ### Histogram ###
    dacBinE = np.arange(dacMin, dacMax + 1, 1)
    hist_max = 10000
    eps = 0.01

    ### Plot ###
    theFig = plt.figure(figsize=(6, 6), facecolor='white')
    mytext = "%s" % (velopix)
    theFig.suptitle(mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

    ### Loop over Data
    mycolors = cm.get_cmap('viridis', len(files) + 1)
    handles = []

    for i in range(len(files)):
        data = np.loadtxt(files[i], dtype=int, delimiter=',')
        hist_data = np.histogram(data, bins=dacBinE)
        logs = hist_data[0].astype(float)
        logs[logs == 108] = eps

        plt.semilogy(hist_data[1][:-1], logs, color=mycolors(i), linestyle='-', linewidth=2)

        words = files[i].split('_')
        handles.append(mpatches.Patch(color=mycolors(i), label=words[-3]))

    plt.legend(bbox_to_anchor=(1.042, 0.8), loc=2, borderaxespad=0, handles=handles, frameon=False, handlelength=1.25)

    ### Axes ###
    plt.axis([dacMin, dacMax, 0.9, hist_max])
    plt.xticks(np.arange(dacMin, dacMax + 1, 100), fontsize=9)

    for tick in plt.subplot(111).yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    plt.xlabel("DAC Threshold", fontsize=15)
    plt.ylabel("Number of Pixels", fontsize=15)

    ### Save ###
    plt.show()
    # plt.savefig(filename+"_Plot_Scans.png", bbox_inches='tight', format='png')


if (len(sys.argv) != 4):
    print("Usage: python plotting_scans.py <file prefix> minThr maxThr")
    print("Example: python plotting_scans.py /home/velo/tmp/Module1_VP0-0 1100 1800")
    exit()

prefix = sys.argv[1]
minThr = int(sys.argv[2])
maxThr = int(sys.argv[3])

plot_scans(prefix, minThr, maxThr)
