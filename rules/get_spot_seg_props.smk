# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given spot segmentation get the object properties table
"""
# =============================================================================
# Params
seg_type = 'spot_seg'
fn_mod = config[seg_type]['fn_mod']

rule get_spot_seg_props:
    input:
        raw_fn = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        seg_fn = (config['output_dir'] + '/' + seg_type + '/{sample_name}'
                  + fn_mod + '.npy')
    output:
        seg_props_fn = (config['output_dir'] + '/' + seg_type + '_props' + '/{sample_name}'
                        + fn_mod + '_props.csv'),
        max_props_fn = (config['output_dir'] + '/' + seg_type + '_props' + '/{sample_name}'
                        + '_max_props.csv'),
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
        "-sp {output.seg_props_fn} "
        "-mp {output.max_props_fn} "
    # conda:
    #     config['conda_env_fn']
