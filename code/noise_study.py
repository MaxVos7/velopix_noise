### Import Modules ###
import sys
import os.path
import math
import glob
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

### Plot Noise Matrix ###
def plot_noiseratio_matrix(filepath, asic, trim, scanA, scanB) :

    ### Load Data ###
    try:
      velopix = asic.split('_')[-1]
      raw_noise_num = np.loadtxt(filepath+scanA+"/"+asic+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      raw_noise_den = np.loadtxt(filepath+scanB+"/"+asic+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      mask = np.loadtxt(filepath+scanB+"/"+asic+"_Matrix_Mask.csv", dtype=int, delimiter=',')
      noise_num = np.nan_to_num(raw_noise_num)
      noise_dem = np.nan_to_num(raw_noise_den)
      noise = np.divide(noise_num, noise_dem, out=np.ones_like(noise_num), where=noise_dem>0)
    except:
      noise = np.ones((256, 256))
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
    
    name = asic_name(velopix)
    mytext = "%s/%s  -  %s  - %0.2f +/- %0.2f" % (name, velopix, trim, mean_n, std_n)
    plt.text(128, 266, mytext, fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    red_patch = mpatches.Patch(color='red', label='N/A')

    plt.legend(bbox_to_anchor=(1.042,0.075), loc=2, borderaxespad=0, handles=[red_patch], frameon=False, handlelength=1.25)
    plt.colorbar(fraction=0.035)
    plt.clim(0.5,2.5)

    # Fix aspect ratio
    plt.axes().set_aspect('equal')

    ### Save ###
    plt.savefig(filepath+asic+"_Plot_Matrix_NoiseRatio_"+scanA+"_"+scanB+"_"+trim+".png", bbox_inches='tight', format='png')


### Plot Noise Histogram ###
def plot_noiseratio_histogram(filepath, asic, trim, scanA, scanB) :

    ### Load Data ###
    try:
      velopix = asic.split('_')[-1]
      raw_noise_num = np.loadtxt(filepath+scanA+"/"+asic+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      raw_noise_den = np.loadtxt(filepath+scanB+"/"+asic+"_"+trim+"_Noise_Width.csv", dtype=float, delimiter=',')
      mask = np.loadtxt(filepath+scanB+"/"+asic+"_Matrix_Mask.csv", dtype=int, delimiter=',')
      noise_num = np.nan_to_num(raw_noise_num)
      noise_dem = np.nan_to_num(raw_noise_den)
      noise = np.divide(noise_num, noise_dem, out=np.ones_like(noise_num), where=noise_dem>0)
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
    name = asic_name(velopix)
    mytext = "%s/%s  -  %s" % (name, velopix, trim)
    theFig.suptitle(mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

    xmin = 0.5
    xmax = 2.5
    edges = [xmin+0.025*i for i in range(0,80)]  # set the x-axis edges of the histogram from xmin to xmax in steps of 0.1
    plt.hist(noise[noise[:]>0].flatten(), bins=edges, histtype='step', color='black')
    plt.hist(odd_noise[odd_noise[:]>0].flatten(), bins=edges, histtype='step', color='blue')
    plt.hist(even_noise[even_noise[:]>0].flatten(), bins=edges, histtype='step', color='red')

    plt.axis([xmin, xmax,0.09,50000])       # set the limits of the x and y axis
    plt.yscale('log', nonposy='clip')       # set the y-axis to log scale
    plt.xticks(np.arange(xmin, xmax+0.1,0.5))
    plt.subplot(111).xaxis.set_ticks(np.arange(xmin, xmax+0.1,0.1), True)
    for tick in plt.subplot(111).yaxis.get_major_ticks():
      tick.label.set_fontsize(15)
    plt.subplot(111).axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)

    mytext = "All:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_n, std_n, min_n, max_n)
    plt.text(0.55, 0.6, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='black')
    mytext = "Odd Col.:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_o, std_o, min_o, max_o)
    plt.text(0.55, 0.3, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='blue')
    mytext = "Even Col.:  %0.2f +/- %0.2f  -  [%0.2f,%0.2f]" % (mean_e, std_e, min_e, max_e)
    plt.text(0.55, 0.15, mytext, fontsize=14, horizontalalignment='left', verticalalignment='center', color='red')

    ### Save ###
    plt.savefig(filepath+asic+"_Plot_Hist_NoiseRatio_"+scanA+"_"+scanB+"_"+trim+".png", bbox_inches='tight', format='png')

### Plot Individual Pixel ###
def plot_pixel():
    #run1_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/default_run1/Module1_VP0-0_ECS_Scan_Trim0_1550_5_90_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    #run1_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/step2_run1/Module1_VP0-0_ECS_Scan_Trim0_1360_2_55_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    run1_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/step1_run1/Module1_VP0-0_ECS_Scan_Trim0_1360_1_110_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    x_run1_trim0 = run1_trim0[:,0]
    y_run1_trim0 = run1_trim0[:,1]
    #run2_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/default_run2/Module1_VP0-0_ECS_Scan_Trim0_1550_5_90_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    #run2_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/step2_run2/Module1_VP0-0_ECS_Scan_Trim0_1360_2_55_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    run2_trim0 = np.loadtxt("/home/velo/kristof/reproducibility/step1_run2/Module1_VP0-0_ECS_Scan_Trim0_1360_1_110_1of1_Pixel_1_1.csv", dtype=int, delimiter=',')
    x_run2_trim0 = run2_trim0[:,0]
    y_run2_trim0 = run2_trim0[:,1]

    run1_array = []
    for i in range(0, len(run1_trim0)):
      for j in range(0, run1_trim0[i,1]):
        run1_array.append(run1_trim0[i,0])
    #run1_hist = np.histogram(run1_array, bins=90, range=(1100,1550))
    print "Run 1:", np.mean(run1_array), np.std(run1_array)

    run2_array = []
    for i in range(0, len(run2_trim0)):
      for j in range(0, run2_trim0[i,1]):
        run2_array.append(run2_trim0[i,0])
    #run2_hist = np.histogram(run2_array, bins=90, range=(1100,1550))
    print "Run 2:", np.mean(run2_array), np.std(run2_array)
    print "Ratio:", np.std(run1_array)/np.std(run2_array)

    ### Plot ###
    theFig = plt.figure(figsize=(8,6), facecolor='white')

    plt.plot(x_run1_trim0, y_run1_trim0, 'ro')
    plt.plot(x_run2_trim0, y_run2_trim0, 'bo')

    plt.axis([1100,1550,0,64])

    ### Save ###
    plt.savefig("pixel_scan.png", bbox_inches='tight', format='png')

### Compare Noise Widths between Scans ###
def compare_noise(filename):

    velopix = filename.split('_')[-1]
    files = sorted(glob.glob(filename+"_ECS_Scan_Trim*_Noise_Fit_Width.csv"))

    scans = [0,1,3,13,15]
    widths0 = []
    widths1 = []
    widths2 = []
    widths3 = []

    for i in range(len(files)):
      data = np.loadtxt(files[i], dtype=float, delimiter=',')
      widths0.append(data[0,0])
      widths1.append(data[0,1])
      widths2.append(data[0,2])
      widths3.append(data[0,3])

    ### Plot ###
    theFig = plt.figure(figsize=(6,6), facecolor='white')
    mytext = "%s" % (velopix)
    theFig.suptitle(mytext, fontsize=20, horizontalalignment='center', verticalalignment='center', color='black')

    plt.fill_between(scans, np.multiply(widths0, 0.95), np.multiply(widths0, 1.05), color='salmon')
    plt.fill_between(scans, np.multiply(widths1, 0.95), np.multiply(widths1, 1.05), color='lightgreen')

    plt.plot(scans, widths0, 'ro', ls='-')
    plt.plot(scans, widths1, 'go', ls='-')
    plt.plot(scans, widths2, 'bo', ls='-')
    plt.plot(scans, widths3, 'co', ls='-')

    ### Save ###
    plt.savefig("width_comparison.png", bbox_inches='tight', format='png')



### Main ###
"""
# Study 1: Ratios
filepath = "/home/velo/kristof/"
for i in range(0,1):
  for j in range(0,1):
    plot_noiseratio_matrix(filepath, "Module1_VP"+str(i)+"-"+str(j), "Trim0", "run1", "run2")
    plot_noiseratio_histogram(filepath, "Module1_VP"+str(i)+"-"+str(j), "Trim0", "run1", "run2")
    plot_noiseratio_matrix(filepath, "Module1_VP"+str(i)+"-"+str(j), "TrimF", "run1", "run2")
    plot_noiseratio_histogram(filepath, "Module1_VP"+str(i)+"-"+str(j), "TrimF", "run1", "run2")

# Study 2: Individual Pixels
plot_pixel()
"""

compare_noise("/home/velo/kristof/fit/trim_scan/Module0_VP0-0")
