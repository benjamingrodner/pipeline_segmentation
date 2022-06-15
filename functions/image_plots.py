import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from copy import copy
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm
from matplotlib import colors
from numba import njit
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries


def get_discrete_colorbar(vals, cmp, integers=True):
    l = max(vals)-min(vals)
    cmp_ = cm.get_cmap(cmp,lut=int(l+1))
    cmp_bounds = np.arange(int(l+2)) - 0.5
    norm = colors.BoundaryNorm(cmp_bounds,cmp_.N)
    image=plt.imshow(np.array([list(vals)]), cmap=cmp_, norm=norm)
    plt.gca().set_visible(False)
    cbar = plt.colorbar(image,ticks=vals,orientation="horizontal")
    if integers:
        cbar.set_ticklabels([str(int(v)) for v in vals])
    else:
        cbar.set_ticklabels([str(v) for v in vals])
    return(cbar)


def subplot_square_images(im_list, subplot_dims, im_inches=5, cmaps=(), clims=(), zoom_coords=(), scalebar_resolution=0, axes_off=True, discrete=()):
    sd1, sd2 = subplot_dims
    figsize=(sd2*im_inches,sd1*im_inches)
    # figsize=(sd2*im_inches,1.02375*sd1*im_inches)
    fig, axes = plt.subplots(sd1,sd2, figsize=figsize)
    for i, (ax, im) in enumerate(zip(fig.axes, im_list)):
        cmap = cmaps[i] if cmaps else 'inferno'
        im_ = im[~np.isnan(im)]
        clim = clims[i] if clims else (np.min(im_), np.max(im_))
        clim = clim if clim else (np.min(im_), np.max(im_))
        if cmap:
            ax.imshow(im, cmap=cmap, clim=clim, interpolation="none")
        else:
            ax.imshow(im, clim=clim)
        zc = zoom_coords if zoom_coords else (0,im.shape[0],0,im.shape[1])
        ax.set_ylim(zc[1],zc[0])
        ax.set_xlim(zc[2],zc[3])
        if axes_off:
            ax.set_axis_off()
        if i == 0 and scalebar_resolution:
            scalebar = ScaleBar(scalebar_resolution, 'um', frameon = False, color = 'white', box_color = 'white')
            plt.gca().add_artist(scalebar)
    plt.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
    discrete = discrete if discrete else np.zeros((len(im_list),))
    cbars = []
    for im, cl, cmp, d in zip(im_list, clims, cmaps, discrete):
        if cmp:
            fig2 = plt.figure(figsize=(9, 1.5))
            if d:
                vals = np.sort(np.unique(im))
                vals = vals[~np.isnan(vals)]
                vals = vals if len(cl)==0 else vals[(vals>=cl[0]) & (vals<=cl[1])]
                cbar = get_discrete_colorbar(vals, d)
            else:
                if cl:
                    image=plt.imshow(im, cmap=cmp, clim=cl)
                else:
                    image=plt.imshow(im, cmap=cmp)
                plt.gca().set_visible(False)
                cbar = plt.colorbar(image,orientation="horizontal")
        cbars.append(cbar)
    return(fig, ax, cbars)


def seg2rgb(seg):
    return label2rgb(seg,  bg_label = 0, bg_color = (0,0,0))

def save_fig(filename, dpi=500):
    plt.savefig(filename, dpi=dpi, bbox_inches='tight',transparent=True)


def _image_figure(dims, dpi=500):
    fig = plt.figure(figsize=(dims[0], dims[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    return(fig, ax)


def plot_image(
            im, im_inches=5, cmap=(), clims=('min','max'), zoom_coords=(), scalebar_resolution=0,
            axes_off=True, discrete=False, cbar_ori='horizontal', dpi=500
        ):
    s = im.shape
    dims = (im_inches*s[1]/np.max(s), im_inches*s[0]/np.max(s))
    fig, ax = _image_figure(dims, dpi=dpi)
    im_ = im[~np.isnan(im)]
    llim = np.min(im_) if clims[0]=='min' else clims[0]
    ulim = np.max(im_) if clims[1]=='max' else clims[1]
    clims = (llim, ulim)
    if cmap:
        ax.imshow(im, cmap=cmap, clim=clims, interpolation="none")
    else:
        ax.imshow(im)
    zc = zoom_coords if zoom_coords else (0,im.shape[0],0,im.shape[1])
    ax.set_ylim(zc[1],zc[0])
    ax.set_xlim(zc[2],zc[3])
    if axes_off:
        ax.set_axis_off()
    if scalebar_resolution:
        scalebar = ScaleBar(
                scalebar_resolution, 'um', frameon = False,
                color = 'white', box_color = 'white'
            )
        plt.gca().add_artist(scalebar)
    cbar = []
    fig2 = []
    if cmap:
        if cbar_ori == 'horizontal':
            fig2 = plt.figure(figsize=(dims[0], dims[0]/10))
        elif cbar_ori == 'vertical':
            fig2 = plt.figure(figsize=(dims[1]/10, dims[1]))
        if discrete:
            vals = np.sort(np.unique(im))
            vals = vals[~np.isnan(vals)]
            vals = vals[(vals>=clims[0]) & (vals<=clims[1])]
            cbar = get_discrete_colorbar(vals, cmap)
        else:
            image=plt.imshow(im, cmap=cmap, clim=clims)
            plt.gca().set_visible(False)
            cbar = plt.colorbar(image,orientation=cbar_ori)
    return(fig, ax, fig2)


def plot_seg_outline(ax, seg, col=(0,1,0)):
    cmap = copy(plt.cm.get_cmap('gray'))
    cmap.set_bad(alpha = 0)
    cmap.set_over(col, 1.0)
    im_line = find_boundaries(seg, mode = 'outer')
    im_line = im_line.astype(float)
    im_line[im_line == 0] = np.nan
    clims = (0,0.9)
    extent = (0,seg.shape[1],0,seg.shape[0])
    ax.imshow(im_line, cmap=cmap, clim=clims, interpolation='none')
    return ax


def get_ROC_curves():
    rd = {}
    # iterate through pos/neg
    for j in J:
        print(i,j)
        sn_fovs = sn_dict[i][j]
        sp_df_all = pd.DataFrame([])
        cell_count = 0
        # Combine fovs
        for k, sn in enumerate(sn_fovs):
            # Get seg
            seg = np.load(seg_dir + '/cell_seg/' + sn[1] + '_cell_seg.npy')
            cell_count += np.unique(seg).shape[0]
            # Get spot params
            sp_df_fn = seg_dir + '/spot_analysis/' + sn[1] + '_max_props_cid.csv'
            sp_df = pd.read_csv(sp_df_fn)
            sp_df['cell_id_fov'] = sp_df.cell_id.astype(str) + '_' + str(k)
            sp_df_all = sp_df_all.append(sp_df)
        # Filter by distance
        sp_df_cell = sp_df_all[(sp_df_all.cell_dist <= max_dist)]
        # Get threshold curves
        # nsrs = [sp_df_cell.loc[(sp_df_cell.intensity < l),'cell_id_fov'] for l in x]
        psrs = [sp_df_cell.loc[(sp_df_cell.intensity >= l),'cell_id_fov'] for l in x]
        rd[j] = {'c':cell_count, 'p':psrs}
        # rd[j] = {'c':cell_count, 'n':nsrs,'p':psrs}
    # calculate values
    # Rs = {}
    # for j in J:
    FPR = [ns.unique().shape[0] / rd['neg']['c'] for ns in rd['neg']['p']]
    TNR = [1-fpr for fpr in FPR]
    TPR = [ns.unique().shape[0] / rd['pos']['c'] for ns in rd['pos']['p']]
    FNR = [1 - tpr for tpr in TPR]
        # Rs[j + '_NR'] = NR
        # Rs[j + '_PR'] = PR
    PPV = [ps.unique().shape[0] / (ps.unique().shape[0] + nps.unique().shape[0])
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    FOR = [(rd['pos']['c'] - ps.unique().shape[0]) / ((rd['pos']['c'] -\
            ps.unique().shape[0]) + (rd['neg']['c'] - nps.unique().shape[0]) + 1e-15)
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    # Save values
    roc_df = pd.DataFrame({'x':x,'TNR':TNR,'FPR':FPR,'FNR':FNR,'TPR':TPR,'PPV':PPV,'FOR':FOR})
    roc_df.to_csv(roc_df_fnt.format(gfn), index=False)


def general_plot(xlabel='', ylabel='', ft=12, dims=(5,3), col='k', lw=1, pad=0):
    fig, ax = plt.subplots(figsize=(dims[0], dims[1]),  tight_layout={'pad': pad})
    for i in ax.spines:
        ax.spines[i].set_linewidth(lw)
    ax.spines['top'].set_color(col)
    ax.spines['bottom'].set_color(col)
    ax.spines['left'].set_color(col)
    ax.spines['right'].set_color(col)
    ax.tick_params(direction='in', labelsize=ft, color=col, labelcolor=col)
    ax.set_xlabel(xlabel, fontsize=ft, color=col)
    ax.set_ylabel(ylabel, fontsize=ft, color=col)
    ax.patch.set_alpha(0)
    return(fig, ax)


def plot_ROC_curves(roc_df, xlims=[], dims=(5,4), thresholds=[],
                    col='k'):
    fig, ax = ip.general_plot(xlabel='Threshold',dims=dims)
    ax.plot(roc_df.x, roc_df.PPV,label='PPV')
    ax.plot(roc_df.x, roc_df.FOR,label='FOR')
    ax.plot(roc_df.x, roc_df.TPR, label='TPR')
    ax.plot(roc_df.x, roc_df.FPR, label='FPR')
    # ax.plot(roc_df.x, roc_df.TNR, label='TNR')
    # ax.plot(roc_df.x, roc_df.FNR,label='FNR')
    ax.plot([threshold]*2, [0,1], col)
    ax.legend()
    ax.set_xlim(xlims)
    return figs
