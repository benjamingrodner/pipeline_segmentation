# Snakefile
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_29
# Last edited : 2022_05_10
# =============================================================================
"""
This pipeline was written to segment  many Airyscan superresolution images of
bacteria stained with 16s rRNA FISH for the cell body and single molecule DNA-
FISH spots, then assign spots to cells
Last edited 2/8/22 BMG
"""
# =============================================================================
import pandas as pd
import os
import sys

# =============================================================================
# Functions
# =============================================================================

def get_input_table(config):
    input_table = pd.read_csv(config['input_table_fn'])
    input_table.columns = config['input_table_cols']
    return input_table

def expand_sn(string):
    return [string.format(sample_name=sn) for sn in SAMPLE_NAMES]

def expand_channels(string, seg_type):
    channels = config[seg_type]['channels']
    return [string.format(sample_name=sn, channel=ch) for ch in channels
            for sn in SAMPLE_NAMES]

def expand_all_channels(string):
    ch_cell = config['cell_seg']['channels']
    ch_spot = config['spot_seg']['channels']
    return [string.format(sample_name=sn, channel_cell=ch_c, channel_spot=ch_s)
            for ch_s in ch_spot
            for ch_c in ch_cell
            for sn in SAMPLE_NAMES]


# =============================================================================
# Parameters
# =============================================================================

args = sys.argv
config_fn = args[args.index("--configfile") + 1]

input_table = get_input_table(config)
SAMPLE_NAMES = input_table['sample_name'].values

csfn = config['cell_seg']['fn_mod']
ssfn = config['spot_seg']['fn_mod']

# Rule all outputs
raw_npy_fns = expand_sn(config['output_dir'] + '/raw_npy/{sample_name}.npy')
cell_seg_fns = expand_channels(config['output_dir'] + '/cell_seg/{sample_name}/'
                            + '{sample_name}_chan_{channel}' + csfn + '.npy',
                            'cell_seg')
spot_seg_fns = expand_channels(config['output_dir'] + '/spot_seg/{sample_name}/'
                            + '{sample_name}_chan_{channel}' + ssfn + '.npy',
                            'spot_seg')
cell_seg_props_fns = expand_channels(config['output_dir'] + '/cell_seg_props/{sample_name}/'
                            + '{sample_name}_chan_{channel}' + csfn + '_props.csv',
                            'cell_seg')
cell_seg_spot_props_fns = expand_all_channels(config['output_dir']
                            + '/cell_seg_props/{sample_name}/'
                            +'{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                            + csfn + '_props.csv')
spot_seg_props_fns = expand_channels(config['output_dir'] + '/spot_seg_props/{sample_name}/'
                            + '{sample_name}_chan_{channel}' + ssfn + '_props.csv',
                            'spot_seg')
spot_props_cid_fns = expand_all_channels(config['output_dir'] + '/spot_analysis/{sample_name}/'
                            + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                            + fn_mod_s + '_props_cid.csv')

# max_props_cid_fns = expand_wildcards(config['output_dir'] + '/spot_analysis/{sample_name}_chan_{channel}_max_props_cid.csv')

# =============================================================================
# Snake rules
# =============================================================================

rule all:
    input:
        cell_seg_props_fns,
        cell_seg_spot_props_fns,
        spot_seg_props_fns,
        spot_props_cid_fns

include: config['pipeline_path'] + '/rules/write_files_to_npy.smk'
include: config['pipeline_path'] + '/rules/segment_cells.smk'
include: config['pipeline_path'] + '/rules/segment_spots.smk'
include: config['pipeline_path'] + '/rules/get_cell_seg_props.smk'
include: config['pipeline_path'] + '/rules/get_cell_seg_spot_props_220608.smk'
include: config['pipeline_path'] + '/rules/get_spot_seg_props_220608.smk'
include: config['pipeline_path'] + '/rules/assign_spots_to_cells_220608.smk'
# include: config['pipeline_path'] + '/rules/assign_spots_to_cells.smk'
