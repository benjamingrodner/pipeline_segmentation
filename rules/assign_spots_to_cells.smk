# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given cell segmentation and spot segmenation properties tables, assign spots to
cells
"""
# =============================================================================
# Params

rule assign_spots_to_cells:
    input:
        cell = (config['output_dir'] + '/cell_seg/{sample_name}_cell_seg.npy'),
        max = (config['output_dir'] + '/spot_seg_props/{sample_name}_max_props.csv')
    output:
         (config['output_dir'] + '/spot_analysis/{sample_name}_max_props_cid.csv')
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
    shell:
        "python {params.pipeline_path}/scripts/assign_spots_to_cells.py "
        "-cfn {params.config_fn} "
        "-cs {input.cell} "
        "-mp {input.max} "
        "-pp {params.pipeline_path} "
        "-mpc {output} "
    # conda:
    #     config['conda_env_fn']
