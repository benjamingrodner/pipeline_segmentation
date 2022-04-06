# Python module
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Contains functions to assign spots segmentation to cell segmentation
"""
# =============================================================================

from skimage.feature import peak_local_max
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from scipy.ndimage.measurements import center_of_mass, label

# =============================================================================


def _get_merged_peaks(im, min_distance=5):
    is_peak = peak_local_max(im, indices=False,
                             min_distance=min_distance) # outputs bool image
    labels = label(is_peak)[0]
    merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    return np.array(merged_peaks)


def group_spot_maxima_by_spot_id(raw, seg, min_distance=5):
    # Get local maxima
    maxs = _get_merged_peaks(raw, min_distance=min_distance)
    # Assign local maxima to spots
    max_ids = [seg[int(m[0]),int(m[1])] for m in maxs]
    intensities = [raw[int(m[0]),int(m[1])] for m in maxs]
    maxs_ = np.concatenate((maxs, np.array([intensities, max_ids]).T),
                                axis=1).tolist()
    # Group maxs by spot id
    maxs_.sort(key=itemgetter(3))
    iter = groupby(maxs_, itemgetter(3))
    maxs_grouped = [[int(key), [v[:3] for v in vals]] for (key, vals) in iter]
    # maxs_grouped = [[id,[(m[0],m[1],m[3]) for m in list(maxs_) if m[2]==id]]
    #                 for id in ids]
    return pd.DataFrame(maxs_grouped, columns=['label', 'maxs_loc_int'])


def get_local_max_props(raw, seg, min_distance=5):
    maxs = _get_merged_peaks(raw, min_distance=min_distance)
    # Assign local maxima to spots
    max_ids = [seg[int(m[0]),int(m[1])] for m in maxs]
    intensities = [raw[int(m[0]),int(m[1])] for m in maxs]
    maxs_ = np.concatenate((maxs, np.array([intensities, max_ids]).T),
                                axis=1).tolist()
    maxs_.sort(key=itemgetter(3))
    return pd.DataFrame(maxs_, columns=['row','col','intensity','spot_id'])



def maxs_to_cell_id(maxs, cell_seg, search_radius):
    cell_seg_pad = np.pad(cell_seg, search_radius, mode='edge')
    maxs_arr = maxs.loc[:,['row','col']].values
    window = 2*search_radius + 1
    max_cid_list = []
    # for each max location
    for i, (r, c) in enumerate(maxs_arr):
        r,c = int(r),int(c)
        cid = cell_seg[r,c]
        # Check if the max is in a cell seg
        if cid:
            dist = 0  # if so distance is zero
        else:
            # get a slice of the seg in a range around the max
            slice = cell_seg_pad[r:r+window, c:c+window]
            cid = 0 # Set non-cell associated values
            dist = 1e5
            for j in range(window):  # iterate over the slice
                for k in range(window):
                    cid_ = slice[j,k]
                    if not cid_:  # Ignore zeros
                        continue
                    else:
                        # calculate the distance to the center of the slice
                        dist_ = ((search_radius-j)**2 + (search_radius-k)**2)**(1/2)
                        # Restrict to a circle (ignore the corners of the slice)
                        if dist_ > search_radius:
                            continue
                        # Replace the distance and cell id values if closer
                        elif dist_ < dist:
                            dist = dist_
                            cid = cid_
        max_cid_list.append([i,cid,dist])  # write max id, cell_id, distance
        max_cid_df = pd.DataFrame(max_cid_list, columns=['max_id','cell_id','cell_dist'])
    return pd.concat([maxs, max_cid_df], axis=1)  # add to max props
