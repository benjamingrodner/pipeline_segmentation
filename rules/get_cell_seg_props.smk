# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given cell segmentation get the object properties table
"""
# =============================================================================
# Params
seg_type = 'cell_seg'
fn_mod = config[seg_type]['fn_mod']

rule get_cell_seg_props:
    input:
        raw_fn = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        seg_fn = (config['output_dir'] + '/' + seg_type + '/{sample_name}'
                  + fn_mod + '.npy')
    output:
        (config['output_dir'] + '/' + seg_type + '_props' + '/{sample_name}' + fn_mod
         + '_props.csv'),
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    shell:
        "python {params.pipeline_path}/scripts/get_seg_props.py "
        "-cfn {params.config_fn} "
        "-r {input.raw_fn} "
        "-s {input.seg_fn} "
        "-st {params.seg_type} "
        "-pp {params.pipeline_path} "
        "-sp {output} "
    # conda:
    #     config['conda_env_fn']
