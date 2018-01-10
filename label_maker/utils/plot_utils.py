#pylint: disable-msg=too-many-arguments
"""
plot_utils.py

@author developmentseed

Code helpful for plotting segmentation model results
"""

from itertools import product

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def _overlay_images(ax, img, mask, vmin, vmax, img_a, mask_a):
    """Overlay intensity mask on top of RGB image (that will be grayscale)"""

    assert img.shape[:2] == mask.shape[:2], 'Arrays must have same width/height'

    im_img = ax.imshow(img, cmap=plt.get_cmap('gray'), alpha=img_a,
                       interpolation='nearest')#, extent=extent)
    im_mask = ax.imshow(mask, cmap=plt.cm.inferno, alpha=mask_a,
                        interpolation='nearest', vmin=vmin, vmax=vmax)#, extent=extent)

    return im_img, im_mask


def _subplot_grid(images, masks, vmin=0, vmax=1, img_a=0.5, mask_a=1):
    """Given a set of images and masks, plot them all overlayed"""

    assert len(images) == len(masks), 'Arrays must have same width/height'

    nrows, ncols = _find_factor_pair(len(images))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    for iter_i, inds in enumerate(product(range(nrows), range(ncols))):
        ax = axes[inds[0], inds[1]]
        _, im_mask = _overlay_images(ax, images[iter_i], masks[iter_i],
                                     vmin, vmax, img_a, mask_a)
        ax.axis('off')

    # Add colorbar to right side of subplots
    fig.subplots_adjust(left=0.025, right=0.9)
    cbar_ax = fig.add_axes([0.905, 0.3, 0.015, 0.4]) # Left, top, width, height
    cbar = fig.colorbar(im_mask, cax=cbar_ax)
    cbar.solids.set_edgecolor('face')
    cbar_ax.set_ylabel('Probability', rotation=270, fontsize=10, labelpad=12)
    cbar_ax.locator_params(nbins=4)
    cbar_ax.tick_params(axis='both', labelsize=8)

    return fig


def _find_factor_pair(val):
    """Return pair of factors that are numerically closest to each other

    Useful for finding dimensions of subplots grid given a number of plots

    Example
    -------
        _find_factor_pair(9) gives 3, 3
        _find_factor_pair(20) gives 4, 5
    """

    factors = []

    for f in range(1, int(val ** 0.5) + 1):
        if val % f == 0:
            factors.append((f, int(val / f)))
    if not factors:
        raise RuntimeError('Could not find non-trival factors for: %s' % val)

    # Factors now contains all factor pairs; find pair closest in value
    diffs = np.diff(np.array(factors), axis=-1)
    diffs[diffs < 0] *= -1  # Only positive vals
    good_ind = np.argmin(diffs)

    good_factors = sorted(factors[good_ind])

    return good_factors


def _convert_imgs_to_gray(rgb_arr):
    """Helper to convert a set of images to grayscale"""
    return np.dot(rgb_arr[..., :3], [0.299, 0.587, 0.114])


def plot_segmentation(images, masks, img_a=0.25, mask_a=1.):
    """Helper to plot images and predicted segmentation masks overlayed

    Parameters
    ----------
    images: numpy.array
        Array containing images with shape:
        (n_images, image_height, img_width, n_channels)
    masks: numpy.array
        Mask predictions. First 3 dimensions must match `images`
    img_a: float
        Alpha value (between 0 and 1) for images. 0 is completely transparent
    mask_a: float
        Alpha value (between 0 and 1) for masks. 0 is completely transparent

    Returns
    -------
    fig: matplotlib.figure
        Figure containing images and segmentation masks overlayed
    """

    # Get prediction percentiles for scaling the colorbar
    vmin, vmax = np.percentile(masks, [0.5, 99.5])

    fig = _subplot_grid(images, masks.squeeze(), vmin, vmax, img_a=img_a,
                        mask_a=mask_a)

    return fig
