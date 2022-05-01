#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

NORM = 'stdnorm' # 'stdnorm', 'featurescaling'
DISTRIBUTION = 7
TYPE = 'known' # 'whole', 'known', 'unknown'
FILL_STRATEGY = 'cero2mean' # 'cero2lesscontact' 'cero2mean'
IMAGE_TYPE = '3d' # '3d', 'concat-horizontal', 'concat-vertical'
FINGERS = 3
LABELS_OUT_FILE = 'labels-' + '-t' + str(DISTRIBUTION) + '-' + FILL_STRATEGY + '-' \
                + NORM + '-' + IMAGE_TYPE + '.npy'
IMAGES_OUT_FILE = 'images-' + '-t' + str(DISTRIBUTION) + '-' + FILL_STRATEGY + '-' \
                + NORM + '-' + IMAGE_TYPE + '.npy'

# Returns the values of the 8 neighbours of a given cell.
# This method is meant to be called with the gaps in tactile cells.
def get_neighbours(tactile_image, cell_x, cell_y):
    pad = 2
    padded_x = cell_x + pad
    padded_y = cell_y + pad

    padded = np.pad(tactile_image, ((pad, pad), (pad, pad)), 'constant') #0s

    neighbours_xs = [padded_x - 1, padded_x - 1, padded_x - 1,
                     padded_x, padded_x,
                     padded_x + 1, padded_x + 1, padded_x + 1]
    neighbours_ys = [padded_y - 1, padded_y, padded_y + 1,
                     padded_y - 1, padded_y + 1,
                     padded_y - 1, padded_y, padded_y + 1]
    num_neighbours = len(neighbours_xs)
    neighbours = []

    for i in range(num_neighbours):
        some_x = neighbours_xs[i]
        some_y = neighbours_ys[i]
        neighbours.append(padded[some_x, some_y])

    return neighbours

def ceros_2_mean(tactile_image):
    prev_tactile_image = np.copy(tactile_image)
    cero_xs, cero_ys = np.where(tactile_image == 0)

    for i in range(len(cero_xs)):
        cell_x = cero_xs[i]
        cell_y = cero_ys[i]
        cell_neighs = get_neighbours(prev_tactile_image, cell_x, cell_y)
        cell_neighs = [value for value in cell_neighs if value > 0.0]

        if len(cell_neighs) > 0:
            tactile_image[cell_x, cell_y] = np.mean(cell_neighs)

    return tactile_image

def create_finger_tactile_image(finger_biotac, normalization, fill_strategy=1):
    tactile_image = np.zeros(shape=(TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
    tactile_image[ELECTRODES_INDEX_ROWS, ELECTRODES_INDEX_COLS] = finger_biotac

    if fill_strategy == 'cero2lesscontact':
        # Strategy 1 - Fill with less contacted value
        # The maximum value corresponds to the less contacted electrode
        max_value = np.max(finger_biotac)
        tactile_image[tactile_image == 0] = max_value
    elif fill_strategy == 'cero2mean':
        # Strategy 2 - Fill with neighbours average
        tactile_image = ceros_2_mean(tactile_image)

        # Repeat in case that there were cells with no values as neighbours, they will now
        if np.min(tactile_image) == 0.0:
            tactile_image = ceros_2_mean(tactile_image)

    if normalization == 'stdnorm':
        tactile_image = (tactile_image - np.mean(tactile_image)) / (np.std(tactile_image))
    elif normalization == 'featurescaling':
        tactile_image = (tactile_image - np.min(tactile_image)) / (np.max(tactile_image) - np.min(tactile_image))

    return tactile_image

def create_grasp_tactile_image(ff_image, mf_image, th_image, image_type='3d'):
    grasp_image = []

    if image_type == '3d':
        grasp_image = np.array([ff_image, mf_image, th_image])
    elif image_type == 'concat-horizontal':
        grasp_image = np.concatenate((ff_image, mf_image, th_image), axis=1)
    elif image_type == 'concat-vertical':
        grasp_image = np.concatenate((ff_image, mf_image, th_image), axis=0)

    return grasp_image

if __name__ == '__main__':

    # Load data
    palmdown_filepath = 'data/biotac-palmdown-grasps.csv'
    palmside_filepath = 'data/biotac-palmside-grasps.csv'
    palmdown_df = pd.read_csv(palmdown_filepath)
    palmside_df = pd.read_csv(palmside_filepath)
    full_df = pd.concat([palmdown_df, palmside_df])
    print(full_df.describe())

    # Plot dataset distribution
    plt.figure()
    full_df['object'].value_counts().plot(kind='bar', figsize=(15, 10), fontsize=15)
    plt.ylabel('grasps', fontsize=15)
    plt.savefig('objects_distribution.eps', format='eps', dpi=300)

    hist = full_df['slipped'].value_counts().plot(kind='bar')

    x_offset = -0.15
    y_offset = 10.00

    for bar in hist.patches:
        b = bar.get_bbox()
        val = "{:+.2f}".format(b.y1 + b.y0)
        hist.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
    plt.savefig('slipped.eps')

    labels = full_df['slipped'].values
    tactiles_df = full_df[['ff_biotac_1', 'ff_biotac_2', 'ff_biotac_3', 'ff_biotac_4', 'ff_biotac_5',
                   'ff_biotac_6', 'ff_biotac_7', 'ff_biotac_8', 'ff_biotac_9', 'ff_biotac_10', 'ff_biotac_11',
                   'ff_biotac_12', 'ff_biotac_13', 'ff_biotac_14', 'ff_biotac_15', 'ff_biotac_16', 'ff_biotac_17',
                   'ff_biotac_18', 'ff_biotac_19', 'ff_biotac_20', 'ff_biotac_21', 'ff_biotac_22', 'ff_biotac_23',
                   'ff_biotac_24', 'mf_biotac_1', 'mf_biotac_2', 'mf_biotac_3', 'mf_biotac_4', 'mf_biotac_5',
                   'mf_biotac_6', 'mf_biotac_7', 'mf_biotac_8', 'mf_biotac_9', 'mf_biotac_10', 'mf_biotac_11',
                   'mf_biotac_12', 'mf_biotac_13', 'mf_biotac_14', 'mf_biotac_15', 'mf_biotac_16', 'mf_biotac_17',
                   'mf_biotac_18', 'mf_biotac_19', 'mf_biotac_20', 'mf_biotac_21', 'mf_biotac_22', 'mf_biotac_23',
                   'mf_biotac_24', 'th_biotac_1', 'th_biotac_2', 'th_biotac_3', 'th_biotac_4', 'th_biotac_5',
                   'th_biotac_6', 'th_biotac_7', 'th_biotac_8', 'th_biotac_9', 'th_biotac_10', 'th_biotac_11',
                   'th_biotac_12', 'th_biotac_13', 'th_biotac_14', 'th_biotac_15', 'th_biotac_16', 'th_biotac_17',
                   'th_biotac_18', 'th_biotac_19', 'th_biotac_20', 'th_biotac_21', 'th_biotac_22', 'th_biotac_23',
                   'th_biotac_24']]
    TACTILE_IMAGE_ROWS = -1
    TACTILE_IMAGE_COLS = -1
    ELECTRODES_INDEX_ROWS = -1
    ELECTRODES_INDEX_COLS = -1
    if DISTRIBUTION == 1:
        TACTILE_IMAGE_ROWS = 8
        TACTILE_IMAGE_COLS = 9
        ELECTRODES_INDEX_ROWS = np.array([0, 1, 3, 3, 4, 4, 4, 5, 6, 7, 0, 1, 3, 3, 4, 4, 4, 5, 6, 7, 1, 2, 2, 3])
        ELECTRODES_INDEX_COLS = np.array([7, 6, 8, 7, 6, 8, 7, 5, 7, 7, 1, 2, 0, 1, 2, 0, 1, 3, 1, 1, 4, 5, 3, 4])
    elif DISTRIBUTION == 2:
        TACTILE_IMAGE_ROWS = 8
        TACTILE_IMAGE_COLS = 7
        ELECTRODES_INDEX_ROWS = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 5, 0, 0, 2, 2, 3, 3, 3, 4, 4, 5, 0, 1, 1, 2])
        ELECTRODES_INDEX_COLS = np.array([6, 5, 6, 5, 4, 6, 5, 4, 5, 5, 0, 1, 0, 1, 2, 0, 1, 2, 1, 1, 3, 4, 2, 3])
    elif DISTRIBUTION == 3:
        TACTILE_IMAGE_ROWS = 6
        TACTILE_IMAGE_COLS = 7
        ELECTRODES_INDEX_ROWS = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 5, 0, 0, 2, 2, 3, 3, 3, 4, 4, 5, 0, 1, 1, 2])
        ELECTRODES_INDEX_COLS = np.array([6, 5, 6, 5, 4, 6, 5, 4, 5, 5, 0, 1, 0, 1, 2, 0, 1, 2, 1, 1, 3, 4, 2, 3])
    elif DISTRIBUTION == 4:
        TACTILE_IMAGE_ROWS = 4
        TACTILE_IMAGE_COLS = 7
        ELECTRODES_INDEX_ROWS = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 1, 1, 2])
        ELECTRODES_INDEX_COLS = np.array([6, 5, 6, 5, 4, 6, 5, 4, 5, 6, 0, 1, 0, 1, 2, 0, 1, 2, 1, 0, 3, 4, 2, 3])
    elif DISTRIBUTION == 5:
        TACTILE_IMAGE_ROWS = 6
        TACTILE_IMAGE_COLS = 5
        ELECTRODES_INDEX_ROWS = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 1, 2])
        ELECTRODES_INDEX_COLS = np.array([4, 4, 4, 3, 3, 4, 4, 3, 3, 4, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 3, 1, 2])
    elif DISTRIBUTION == 6:
        TACTILE_IMAGE_ROWS = 6
        TACTILE_IMAGE_COLS = 5
        ELECTRODES_INDEX_ROWS = np.array([0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 1, 2])
        ELECTRODES_INDEX_COLS = np.array([4, 3, 4, 3, 3, 4, 4, 3, 3, 4, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 2, 3, 1, 2])
    elif DISTRIBUTION == 7:
        TACTILE_IMAGE_ROWS = 12
        TACTILE_IMAGE_COLS = 11
        ELECTRODES_INDEX_ROWS = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 3, 5])
        ELECTRODES_INDEX_COLS = np.array([1, 2, 0, 1, 3, 0, 1, 4, 2, 1, 9, 8, 10, 9, 7, 10, 9, 6, 8, 9, 5, 4, 6, 5])
    if IMAGE_TYPE == '3d':
        tactile_images = np.zeros(shape=(tactiles_df.shape[0], FINGERS, TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
    elif IMAGE_TYPE == 'concat-horizontal':
        tactile_images = np.zeros(shape=(tactiles_df.shape[0], 1, TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS * FINGERS))
    elif IMAGE_TYPE == 'concat-vertical':
        tactile_images = np.zeros(shape=(tactiles_df.shape[0], 1, TACTILE_IMAGE_ROWS * FINGERS, TACTILE_IMAGE_COLS))

    for sample in range(tactiles_df.shape[0]):
        one_grasp = tactiles_df.iloc[sample].values

        ff_image = create_finger_tactile_image(one_grasp[0:24], normalization=NORM, fill_strategy=FILL_STRATEGY)
        mf_image = create_finger_tactile_image(one_grasp[24:48], normalization=NORM, fill_strategy=FILL_STRATEGY)
        th_image = create_finger_tactile_image(one_grasp[48:], normalization=NORM, fill_strategy=FILL_STRATEGY)
        tactile_images[sample] = create_grasp_tactile_image(ff_image, mf_image, th_image, image_type=IMAGE_TYPE)

    some_grasp = 0
    print(labels[some_grasp])
    print(tactile_images[some_grasp])

    full_labels = labels
    full_images = tactile_images

    np.save(LABELS_OUT_FILE, arr=full_labels)
    np.save(IMAGES_OUT_FILE, arr=full_images)
    print(full_labels[3:])

    unique = int(full_images.shape[0] / 5)
    index = unique + 1

    print(unique)

    plt.figure()
    plt.imshow(full_images[0 * unique + index, 0, :, :], interpolation='nearest', )
    plt.axis('off')
    plt.title('original')
    plt.colorbar()

    plt.figure()
    plt.imshow(full_images[1 * unique + index, 0, :, :], interpolation='nearest', )
    plt.axis('off')
    plt.title('fliplr')
    plt.colorbar()

