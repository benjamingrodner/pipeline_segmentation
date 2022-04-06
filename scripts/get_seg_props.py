# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given a segmented image, get the object properties table
"""
# =============================================================================

import yaml
import sys
import numpy as np
import pandas as pd
import argparse

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cfn', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-r', '--raw_fn', dest ='raw_fn', type = str, help = '')
    parser.add_argument('-s', '--seg_fn', dest ='seg_fn', type = str, help = '')
    parser.add_argument('-st', '--seg_type', dest ='seg_type', type = str, help = '')
    parser.add_argument('-pp', '--pipeline_path', dest ='pipeline_path', type = str, help = '')
    parser.add_argument('-sp', '--seg_props_fn', dest ='seg_props_fn', type = str, help = '')
    parser.add_argument('-mp', '--max_props_fn', dest ='max_props_fn', default='', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    pdict = config[args.seg_type]

    # Load segmentation and raw image
    raw = np.load(args.raw_fn)
    seg = np.load(args.seg_fn)

    # get 2d raw image
    sys.path.append(args.pipeline_path + '/functions')
    import segmentation_func as sf
    raw_2D = sf.max_projection(raw, pdict['channels'])

    # Get properties table
    print('LOOK HERE \n\n', seg.shape, raw_2D.shape)
    props = sf.measure_regionprops(seg, raw_2D)

    # add maxima locations table
    if pdict['maxima']:
        import spot_funcs as spf
        # # formerly: assign maxs to spot seg id and merge with spot props table
        # maxs = spf.group_spot_maxima_by_spot_id(raw_2D, seg,
        #                               min_distance=config['local_max_mindist'])
        # props = props.merge(maxs, how='left',on='label')

        # mask the raw image using the seg
        raw_2D_masked = raw_2D*(seg>0)
        maxs = spf.get_local_max_props(raw_2D_masked, seg,
                                       min_distance=config['local_max_mindist'])
        # write a separate local max table
        maxs.to_csv(args.max_props_fn, index=False)

    # Save props table
    props.to_csv(args.seg_props_fn, index=False)

    return

if __name__ == '__main__':
    main()
