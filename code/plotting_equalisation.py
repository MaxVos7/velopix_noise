### Import Modules ###
import sys
import os.path
import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

### ASIC mapping ###
def asic_name(velopix):
    tile = int(velopix[2])
    vpx = str(velopix[4])
    if (tile==0):
      return "CLI-"+vpx
    elif (tile==1):
      return "NLO-"+vpx
    elif (tile==2):
      return "NSI-"+vpx
    elif (tile==3):
      return "CSO-"+vpx
    else:
      return "N/A"

### Plot Scan Summary ###
def plot_scan(filename, dacMin, dacMax):

  addCont = os.path.isfile(filename+"_Control_Noise_Mean.csv")

  ### Load Data ###
  try:
    velopix = filename.split('_')[-1]
    mask = np.loadtxt(filename+"_Matrix_Mask.csv", dtype=int, delimiter=',')
    data_Trim0 = np.loadtxt(filename+"_Trim0_Noise_Mean.csv", dtype=float, delimiter=',')
    data_TrimF = np.loadtxt(filename+"_TrimF_Noise_Mean.csv", dtype=float, delimiter=',')
    data_Equal = np.loadtxt(filename+"_TrimBest_Noise_Predict.csv", dtype=int, delimiter=',')
    if (addCont):
      data_Control = np.loadtxt(filename+"_Control_Noise_Mean.csv", dtype=int, delimiter=',')
  except:
    print("Error loading files")
    pass
    
  nMasked = np.count_nonzero(mask)

  ### Turn into Histogram ###
  dacBinE = np.arange(dacMin, dacMax+1, 1)

  hist_data_Trim0 = np.histogram(data_Trim0, bins=dacBinE)
  hist_data_TrimF = np.histogram(data_TrimF, bins=dacBinE)
  hist_data_Equal = np.histogram(data_Equal, bins=dacBinE)
  if (addCont):
    hist_data_Control = np.histogram(data_Control, bins=dacBinE)
  # compute the highest value rounded to nearest 1000
  hist_max = math.ceil((np.amax(hist_data_Equal[0])+1000)/1000)*1000

  target = np.mean( 0.5*(data_Trim0[(data_Trim0>0) & (data_TrimF>0)]+data_TrimF[(data_Trim0>0) & (data_TrimF>0)]) )

  ### Plot ###
  theFig = plt.figure(figsize=(6,6), facecolor='white')
  asic = asic_name(velopix)
  mytext = "%s/%s" % (asic, velopix)
  theFig.suptitle(mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

  eps = 0.01
  # convert zeros to small
  logs = hist_data_Trim0[0].astype(float)
  logs[logs==0] = eps
  plt.semilogy(hist_data_Trim0[1][:-1], logs, color='red', linestyle='-', linewidth=2)
  logs = hist_data_TrimF[0].astype(float)
  logs[logs==0] = eps
  plt.semilogy(hist_data_TrimF[1][:-1], logs, color='blue', linestyle='-', linewidth=2)
  logs = hist_data_Equal[0].astype(float)
  logs[logs==0] = eps
  plt.semilogy(hist_data_Equal[1][:-1], logs, color='black', linestyle='-', linewidth=2)
  if (addCont):
    logs = hist_data_Control[0].astype(float)
    logs[logs==0] = eps
    plt.semilogy(hist_data_Control[1][:-1], logs, color='green', linestyle='-', linewidth=2)

  ### Axes ###
  plt.axis([dacMin,dacMax,0.9,hist_max])
  plt.xticks(np.arange(dacMin,dacMax+1,100), fontsize=9)
  plt.subplot(111).xaxis.set_ticks(np.arange(dacMin, dacMax+1,20), True)

  for tick in plt.subplot(111).yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
  plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

  plt.xlabel("DAC Threshold", fontsize=15)
  plt.ylabel("Number of Pixels", fontsize=15)

  # Stats
  mytext = "0 Trim:\n%.1f +/- %.1f" % (np.mean(data_Trim0[data_Trim0>0]), np.std(data_Trim0[data_Trim0>0]))
  plt.text(dacMax + 20, math.exp(8), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='red')
  mytext = "F Trim:\n%.1f +/- %.1f" % (np.mean(data_TrimF[data_TrimF>0]), np.std(data_TrimF[data_TrimF>0]))
  plt.text(dacMax + 20, math.exp(7), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='blue')
  mytext = "Target:\n%.1f" % target
  plt.text(dacMax + 20, math.exp(6), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
  mytext = "Predicted:\n%.1f +/- %.1f" % (np.mean(data_Equal[data_Equal>0]), np.std(data_Equal[data_Equal>0]))
  plt.text(dacMax + 20, math.exp(5), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
  if (addCont):
    mytext = "Achieved:\n%.1f +/- %.1f" % (np.mean(data_Control[data_Control>0]), np.std(data_Control[data_Control>0]))
    plt.text(dacMax + 20, math.exp(4), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='green')

  mytext = "Masked:\n%d" % (nMasked)
  plt.text(dacMax + 20, math.exp(2), mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')

  ### Save ###
  plt.savefig(filename+"_Plot_Summary.png", bbox_inches='tight', format='png')


### Plot Mask Matrix ###
# Cat A: No response from Either scan (dead pixel)
# Cat B: No response from Trim0 scan (out of range?)
# Cat C: No response from TrimF scan (out of range?)
# Cat D: Target cannot be reached with trim between 0 and 15
# Cat E: Predicted noise mean outside of acceptance window
def plot_mask(filename):

  ### Load Data ###
  try:
    velopix = filename.split('_')[-1]
    mask = np.loadtxt(filename+"_Matrix_Mask.csv", dtype=int, delimiter=',')
  except:
    #occurs if the file doesn't exist
    velopix = "No File"
    mask = np.zeros((256,256))-1

  nMasked = np.count_nonzero(mask)

  ### Set the colours ###
  bitmap = np.zeros((256,256,3)) # create a 256x256x3 array where each pixel has an RGB value of 1-bit representation
  bitmap[mask==1] = [0,0,0] # Cat A: black
  bitmap[mask==2] = [1,0,0] # Cat B: red
  bitmap[mask==3] = [0,0,1] # Cat C: blue
  bitmap[mask==4] = [0,0.502,0] # Cat D: green
  bitmap[mask==5] = [1,0.647,0] # Cat E: orange -> from matplotlib.color.to_rgba("orange")
  bitmap[mask==0] = [0.9,0.9,0.9] # No mask: light grey

  ### Plot ###
  theFig = plt.figure(figsize=(6,6), facecolor='white')
  plt.imshow(bitmap, aspect=1, origin='lower')
  plt.axis([-2,257,-2,257]) # to better visualise the border
  plt.xticks(np.arange(0,257,16), fontsize=8)
  plt.yticks(np.arange(0,257,16), fontsize=8)
  
  asic = asic_name(velopix)
  mytext = "%s/%s - Masked %d pixels" % (asic, velopix, nMasked)
  plt.text(128, 266, mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')
  
  mytext = "Cat A: %d" % (np.count_nonzero(mask==1))
  catA = mpatches.Patch(color='black', label=mytext)
  mytext = "Cat B: %d" % (np.count_nonzero(mask==2))
  catB = mpatches.Patch(color='red', label=mytext)
  mytext = "Cat C: %d" % (np.count_nonzero(mask==3))
  catC = mpatches.Patch(color='blue', label=mytext)
  mytext = "Cat D: %d" % (np.count_nonzero(mask==4))
  catD = mpatches.Patch(color='green', label=mytext)
  mytext = "Cat E: %d" % (np.count_nonzero(mask==5))
  catE = mpatches.Patch(color='orange', label=mytext)
  mytext = "Tot: %d" % (nMasked)
  catT = mpatches.Patch(color='white', label=mytext)
  plt.legend(bbox_to_anchor=(1.042,0.3), loc=2, borderaxespad=0, handles=[catA,catB,catC,catD,catE,catT], frameon=False, handlelength=1.25)

  # Fix aspect ratio
  plt.axes().set_aspect('equal')

  ### Save ###
  plt.savefig(filename+"_Plot_Matrix_Mask.png", bbox_inches='tight', format='png')


### Plot Trim Matrix ###
def plot_trim(filename):

    ### Load Data ###
    try:
      velopix = filename.split('_')[-1]
      trim = np.loadtxt(filename+"_Matrix_Trim.csv", dtype=int, delimiter=',')
      mask = np.loadtxt(filename+"_Matrix_Mask.csv", dtype=int, delimiter=',')
    except:
      #occurs if the file doesn't exist
      velopix = "No File"
      trim = np.zeros((256,256))-1
      mask = np.zeros((256, 256)) - 1

    mtrim = np.ma.masked_where(mask>0, trim)

    ### Plot ###
    theFig = plt.figure(figsize=(6,6), facecolor='white')

    cMap = plt.cm.viridis
    cMap.set_bad('red', 1.0)

    plt.imshow(mtrim, aspect=1, cmap=cMap, origin='lower')
    plt.axis([-2,257,-2,257]) # to better visualise the border
    plt.xticks(np.arange(0,257,16), fontsize=8)
    plt.yticks(np.arange(0,257,16), fontsize=8)
    
    asic = asic_name(velopix)
    mytext = "%s/%s" % (asic, velopix)
    plt.text(128, 266, mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

    red_patch = mpatches.Patch(color='red', label='N/A')

    plt.legend(bbox_to_anchor=(1.042,0.075), loc=2, borderaxespad=0, handles=[red_patch], frameon=False, handlelength=1.25)
    plt.colorbar(fraction=0.035)
    plt.clim(0,15)

    # Fix aspect ratio
    plt.axes().set_aspect('equal')

    ### Save ###
    plt.savefig(filename+"_Plot_Matrix_Trim.png", bbox_inches='tight', format='png')


### Plot Noise Matrix ###
def plot_noise_matrix(filename, trim) :

    ### Load Data ###
    try:
      velopix = filename.split('_')[-1]
      raw_noise = np.loadtxt(filename+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      mask = np.loadtxt(filename+"_Matrix_Mask.csv", dtype=int, delimiter=',')
      noise = np.nan_to_num(raw_noise)
    except:
      noise = np.zeros((256, 256))
      mask  = np.ones((256, 256))

    mnoise = np.ma.masked_where(mask>0, noise)

    ### Stats ###
    max_n = np.max(mnoise)
    min_n = np.min(mnoise)
    mean_n = np.mean(mnoise)
    std_n = np.std(mnoise)

    ### Plot ###
    theFig = plt.figure(figsize=(6,6), facecolor='white')

    cMap = plt.cm.viridis
    cMap.set_bad('red', 1.0)

    plt.imshow(mnoise, aspect=1, cmap=cMap, origin='lower')
    plt.axis([-2,257,-2, 257]) # to better visualise the border
    plt.xticks(np.arange(0,257,16), fontsize=8)
    plt.yticks(np.arange(0,257,16), fontsize=8)
    
    asic = asic_name(velopix)
    mytext = "%s/%s  -  %s  - %0.2f +/- %0.2f" % (asic, velopix, trim, mean_n, std_n)
    plt.text(128, 266, mytext, fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    red_patch = mpatches.Patch(color='red', label='N/A')

    plt.legend(bbox_to_anchor=(1.042,0.075), loc=2, borderaxespad=0, handles=[red_patch], frameon=False, handlelength=1.25)
    plt.colorbar(fraction=0.035)
    plt.clim(4,8)

    # Fix aspect ratio
    plt.axes().set_aspect('equal')

    ### Save ###
    plt.savefig(filename+"_Plot_Matrix_Noise_"+trim+".png", bbox_inches='tight', format='png')

### Plot Noise Histogram ###
def plot_noise_histogram(filename, trim) :

    ### Load Data ###
    try:
      velopix = filename.split('_')[-1]
      raw_noise = np.loadtxt(filename+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      mask = np.loadtxt(filename+"_Matrix_Mask.csv", dtype=int, delimiter=',')
      noise = np.nan_to_num(raw_noise)
    except:
      noise = np.zeros((256, 256))
      mask  = np.ones((256, 256))

    mnoise = np.ma.masked_where(mask>0, noise)

    max_n = np.max(mnoise)
    min_n = np.min(mnoise)
    mean_n = np.mean(mnoise)
    std_n = np.std(mnoise)

    odd_noise = mnoise[:, 0::2]
    max_o = np.max(odd_noise)
    min_o = np.min(odd_noise)
    mean_o = np.mean(odd_noise)
    std_o = np.std(odd_noise)

    even_noise = mnoise[:, 1::2]
    max_e = np.max(even_noise)
    min_e = np.min(even_noise)
    mean_e = np.mean(even_noise)
    std_e = np.std(even_noise)

    ### Plot ###
    theFig = plt.figure(figsize=(8,6), facecolor='white')
    asic = asic_name(velopix)
    mytext = "%s/%s  -  %s" % (asic, velopix, trim)
    theFig.suptitle(mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

    xmin = 4
    xmax = 10
    edges = [i/10.0 for i in range(10*xmin,10*xmax+1)]  # set the x-axis edges of the histogram from xmin to xmax in steps of 0.1
    plt.hist(noise[noise[:]>0].flatten(), bins=edges, histtype='step', color='black')
    plt.hist(odd_noise[odd_noise[:]>0].flatten(), bins=edges, histtype='step', color='blue')
    plt.hist(even_noise[even_noise[:]>0].flatten(), bins=edges, histtype='step', color='red')

    plt.axis([xmin, xmax,0.09,10000])       # set the limits of the x and y axis
    plt.yscale('log', nonposy='clip')       # set the y-axis to log scale
    plt.xticks(np.arange(xmin, xmax+1,1))
    plt.subplot(111).xaxis.set_ticks(np.arange(xmin, xmax+0.2,0.2), True)
    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    mytext = "All:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_n, std_n, min_n, max_n)
    plt.text(4.25, 0.6, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='black')
    mytext = "Odd Col.:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_o, std_o, min_o, max_o)
    plt.text(4.25, 0.3, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='blue')
    mytext = "Even Col.:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_e, std_e, min_e, max_e)
    plt.text(4.25, 0.15, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='red')

    ### Save ###
    plt.savefig(filename+"_Plot_Hist_Noise_"+trim+".png", bbox_inches='tight', format='png')

if (len(sys.argv)!=4):
    print "Usage: python plotting_equalisation.py <file prefix> minThr maxThr"
    print "Example: python plotting_equalisation.py /home/velo/tmp/Module1_VP0-0 1100 1800"
    exit

prefix = sys.argv[1]
minThr = int(sys.argv[2])
maxThr = int(sys.argv[3])

plot_scan(prefix, minThr, maxThr)
plot_mask(prefix)
plot_trim(prefix)
plot_noise_matrix(prefix, "Trim0")
plot_noise_histogram(prefix, "Trim0")
plot_noise_matrix(prefix, "TrimF")
plot_noise_histogram(prefix, "TrimF")

