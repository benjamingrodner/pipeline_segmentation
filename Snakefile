# Snakefile
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_29
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

# =============================================================================
# Parameters
# =============================================================================

args = sys.argv
config_fn = args[args.index("--configfile") + 1]

input_table = get_input_table(config)
SAMPLE_NAMES = input_table['sample_name'].values
INPUT_FILENAMES_FULL = [config['input_dir'] + '/' + sn + config['input_ext']
                        for sn in SAMPLE_NAMES]

# Rule all outputs
raw_npy_fns = expand_sn(config['output_dir'] + '/raw_npy/{sample_name}.npy')
cell_seg_fns = expand_sn((config['output_dir'] + '/cell_seg/{sample_name}'
                            + config['cell_seg']['fn_mod'] + '.npy'))
spot_seg_fns = expand_sn((config['output_dir'] + '/spot_seg/{sample_name}'
                            + config['spot_seg']['fn_mod'] + '.npy'))
cell_seg_props_fns = expand_sn((config['output_dir'] + '/cell_seg_props/{sample_name}'
                                + config['cell_seg']['fn_mod'] + '_props.csv'))
spot_seg_props_fns = expand_sn((config['output_dir'] + '/spot_seg_props/{sample_name}'
                                + config['spot_seg']['fn_mod'] + '_props.csv'))
max_props_cid_fns = expand_sn((config['output_dir'] + '/spot_analysis/{sample_name}_max_props_cid.csv'))

# =============================================================================
# Snake rules
# =============================================================================
wildcard_constraints:
    sample_name=config['input_regex']
rule all:
    input:
        cell_seg_props_fns,
        max_props_cid_fns

# include: 'rules/write_files_to_npy.smk'
include: 'rules/segment_cells.smk'
include: 'rules/segment_spots.smk'
include: 'rules/get_cell_seg_props.smk'
include: 'rules/get_spot_seg_props.smk'
include: 'rules/assign_spots_to_cells.smk'
