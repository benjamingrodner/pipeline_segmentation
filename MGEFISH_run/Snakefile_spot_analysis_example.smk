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

def expand_wildcards(string):
    return [string.format(sample_name=sn, channel=ch) for ch in SPOT_CHANNELS
            for sn in SAMPLE_NAMES]


# =============================================================================
# Parameters
# =============================================================================

args = sys.argv
config_fn = args[args.index("--configfile") + 1]

input_table = get_input_table(config)
SAMPLE_NAMES = input_table['sample_name'].values
INPUT_FILENAMES_FULL = [config['input_dir'] + '/' + sn + config['input_ext']
                        for sn in SAMPLE_NAMES]
SPOT_CHANNELS = config['spot_seg']['channels']

# Rule all outputs
max_props_cid_fns = expand_wildcards(config['output_dir'] + '/spot_analysis/{sample_name}_chan_{channel}_max_props_cid.csv')

# =============================================================================
# Snake rules
# =============================================================================

rule all:
    input:
        max_props_cid_fns

# include: config['pipeline_dir'] + '/rules/filter_spots.smk'
include: config['pipeline_dir'] + '/rules/assign_spots_to_cells_220608.smk'
