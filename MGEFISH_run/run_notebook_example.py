# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_28
# =============================================================================
"""
The notebook Has Been Built for...Running the segmentation_pipeline

For use with the 'hiprfish_imaging_py38' conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
# Modify
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/mobile_elements/experiments/2022_01_22_gelgfp'
                    # Absolute path to the project work directory
config_fn = 'config.yaml' # relative path to config file from workdir

# %% codecell
# Imports
import glob
import pandas as pd
import subprocess
import yaml
import gc
import os
import javabridge
import bioformats
import sys
import numpy as np
from matplotlib.cm import get_cmap

# %% codecell
# Set up notebook stuff
%load_ext autoreload
%autoreload 2
gc.enable()

# %% codecell
# Move to notebook directory
os.chdir(project_workdir)
os.getcwd()

# %% codecell
# load config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% codecell
# Function imports
sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf

# %% codecell
# =============================================================================
# Get test image
# =============================================================================
#  get input files automatically
input_filenames = glob.glob(config['input_dir'] + '/*' + config['input_ext'])
input_filenames.sort()
print(len(input_filenames))
input_filenames

# %% codecell
# Select one image to set parameters
test_index = 11
input_fn = input_filenames[test_index]
input_fn

# %% codecell
# load the image
javabridge.start_vm(class_path=bioformats.JARS)
input = bioformats.load_image(input_fn)
input.shape

# %% codecell
# show the image
clims = [(),(0,0.05)]
n_channels = input.shape[2]
im_list = [input[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% codecell
# subset the image
sr = [(300,500),(250,450)]
input_sub = input[sr[0][0]:sr[0][1],sr[1][0]:sr[1][1]]
im_list = [input_sub[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% codecell
# =============================================================================
# Set cell segmetnation parameters
# =============================================================================
# get cell channel
im_cell_list = [input_sub[:,:,i] for i in config['cell_seg']['channels']]
if len(im_cell_list) > 1:
    im_cell = np.max((np.dstack(im_cell_list)), axis=2)
else:
    im_cell = im_cell_list[0]

# %% codecell
# set cell segmentation parameters in config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

# %% codecell
# run segmentation
im_cell_pre = sf.pre_process(
    im_cell,
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss']
    )
im_cell_mask = sf.get_background_mask(
    im_cell,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
im_cell_seg = sf.segment(
    im_cell_pre,
    background_mask = im_cell_mask,
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )
im_cell_seg.shape

# %% codecell
# show seg process
seg_rgb = ip.seg2rgb(im_cell_seg)
im_list = [im_cell, im_cell_pre, im_cell_mask, seg_rgb]
ip.subplot_square_images(im_list, (1,4))

# %% codecell
# =============================================================================
# Set spot segmetnation parameters
# =============================================================================
# get cell channel
im_spot_list = [input_sub[:,:,i] for i in config['spot_seg']['channels']]
if len(im_spot_list) > 1:
    im_spot = np.max((np.dstack(im_spot_list)), axis=2)
else:
    im_spot = im_spot_list[0]

# %% codecell
# set cell segmentation parameters in config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

# %% codecell
# run segmentation
im_spot_pre = sf.pre_process(
    im_spot,
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss']
    )
im_spot_mask = sf.get_background_mask(
    im_spot,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
im_spot_seg = sf.segment(
    im_spot_pre,
    background_mask = im_spot_mask,
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )
im_spot_seg.shape

# %% codecell
# show seg process
seg_rgb = ip.seg2rgb(im_spot_seg)
im_list = [im_spot, im_spot_pre, im_spot_mask, seg_rgb]
ip.subplot_square_images(im_list, (1,4))


# %% codecell
# =============================================================================
# Set spot max parameters
# =============================================================================
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# maxs = spf.peak_local_max(im_spot, min_distance = config['local_max_mindist'])
ma = spf._get_merged_peaks(im_spot, min_distance=config['local_max_mindist'])
# is_peak = spf.peak_local_max(im_spot, indices=False, min_distance=config['local_max_mindist']) # outputs bool image
# is_peak.shape
# labels = spf.label(is_peak)[0]
# merged_peaks = spf.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
# ma = np.array(merged_peaks)
fig, ax, cbar = ip.plot_image(im_spot, cmap='inferno', im_inches=20)
ax.scatter(ma[:,1],ma[:,0], s=50, color=(0,1,0))
ax = ip.plot_seg_outline(ax, im_spot_seg, col=(0,0.8,0.8))
# ax.set_xlim((300,400))
# ax.set_ylim((425,475))

# %% codecell
# =============================================================================
# Run the seg pipeline
# =============================================================================

# %% codecell
# Write the test input_table
input_fns_split = [os.path.split(fn)[1] for fn in [input_fn]]
sample_names = [os.path.splitext(fn)[0] for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table


# %% codecell
# Write the full input_table
input_fns_split = [os.path.split(fn)[1] for fn in input_filenames]
sample_names = [os.path.splitext(fn)[0] for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table

# %% codecell
# Execute the snakemake
dry_run = False  # Just create DAG if True
n_cores = 1  # number of allowed cores for the snakemake to use
force_run = 'get_spot_seg_props'  # False if none

snakefile = config['pipeline_path'] + '/' + config['snakefile']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = [
    'snakemake', '-s', snakefile, '--configfile', config_fn, '-j', str(n_cores),
    dr, fr
    ]
print(" ".join(command))
subprocess.check_call(command)

# %% codecell
# =============================================================================
# Select a spot threshold
# =============================================================================
# Load max props table and get intensity values
factors = ['moi','time','fov']
keys = [imfn.get_filename_keys(sn, factors) for sn in sample_names]
sn_dict = imfn.get_nested_dict(keys, sample_names, [0,1])

# %% codecell
# Get threshold curves
x = np.linspace(0,0.1,20)  # thresholds
max_props_fnt = (config['output_dir'] + '/' + 'spot_seg_props' + '/{sample_name}/'
                + '{sample_name}_chan_{channel_spot}' + fn_mod + '_max_props.csv')
curve_dict = {}
for cs in config['spot_seg']['channels']:
    curve_dict[cs] = {}
    for g in groups:
        curve_dict[cs][g] = {}
        for sg in subgroups:
            sn_fovs = sn_dict[g][sg]
            col = colors_tab20[c]
            for sn in sn_fovs:
                max_props = max_props_fnt.format(sample_name=sn, channel_spot=cs)
                curves.append([max_props[max_props.intensity > t].shape[0]
                                for t in x])
            curve_dict[cs][g][sg] = curves

# %% codecell
# plot threshold curves
filt_vals = [0.05,0.05]  # adjust
xlims = [(0,0.1),(0,0.1)]
lw = 0.5
alpha = 0.8
colors_tab20 = get_cmap('tab20').colors
thresh_curve_fnt = config['spot_filter']['filter_dir']
                    + '/thresh_curves_chan_{}'
c = 0
for cs, xlms, fv in zip(config['spot_seg']['channels'], xlims, filt_vals):
    fig, ax = ip.general_plot(xlabel='Threshold',dims=dims)
    for g in groups:
        for sg in subgroups:
            curve_list = curve_dict[cs][g][sg]
            col = colors_tab20[c]
            for crv in curve_list:
                ax.plot(x, crv, c=col, lw=lw, alpha=alpha)
        c += 1
    ip.plt.legend()
    ip.plt.set_xlims(xlms)
    ymax = ax.get_ylim()[1]
    ax.plot([fv]*2, [0,ymax], '-k')
    ip.save_png_pdf(thresh_curve_fnt.format(cs))


# %% codecell
# pick threholds and save to file
filt_vals_dict = {zip(config['spot_seg']['channels'],filt_vals)}
filt_vals_df = pd.DataFrame(filt_vals_dict)
filt_vals_df.to_csv(config['filter_values_fn'])



# %% clims_cell
