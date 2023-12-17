from pathlib import Path
import cv2
import numpy as np
from matplotlib import tri as tri
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from PIL import Image, ImageDraw


def plot_data(triangles, coordinates):
    grid_x, grid_y = np.mgrid[1:99:100j, 1:49:50j]

    cellcenters = np.zeros((triangles.shape[0], 2))

    for cell in range(triangles.shape[0]):
        verts = triangles[cell]
        cellcenter = np.sum(coordinates[verts], axis=0) / 3
        cellcenters[cell, :] = cellcenter[:2]

    return (cellcenters, grid_x, grid_y)

def create_img(plot_data, values):
    cellcenters, grid_x, grid_y = plot_data

    # grid_z = griddata(cellcenters, values, (grid_x, grid_y), method='nearest')
    grid_z = griddata(cellcenters, values, (grid_x, grid_y), method='linear')
    # grid_z = griddata(cellcenters, values, (grid_x, grid_y), method='cubic')

    return grid_z.T


def detect_dryspots(img, img_save_path=None):
    # igonre first 5 cm of the image, because dryspots here are unrealistic
    img = np.copy(img[:, :])  
    img[:, :5] = 255

    _, gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # If stuff at the edges might not be detected you can try the following code:
    # https://answers.opencv.org/question/213719/simpleblobdetector_params-doesnt-detect-the-blobs-on-the-edges-of-the-image/

    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 65
    params.maxThreshold = 93
    params.blobColor = 0
    params.minArea = 5
    params.maxArea = 500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity = .2
    params.maxCircularity = 1

    det = cv2.SimpleBlobDetector_create(params)
    keypts = det.detect(gray)

    im_with_keypoints = cv2.drawKeypoints(gray, keypts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if img_save_path is not None:
        cv2.imwrite(img_save_path / "dry_spot_detection.png", im_with_keypoints)

    if len(keypts) > 0:
        dryspot = True
    else:
        dryspot = False

    return dryspot


def get_leftmost_and_rightmost_edges_of_ff_min_max(img, img_save_path=None):
    img = img[:, :]

    # a = np.where(img_gray > 30, 255, img_gray)
    # black_and_white = np.where(a <= 30, 0, a)

    # Apply edge detection method on the image
    edges2 = cv2.Canny(img, 30, 40, apertureSize = 3)

    right_most_positions = []
    left_most_positions = []
    for e in edges2:
        arr = np.argwhere(e==255).flatten()
        if arr.any() == False:
            continue
        right_most_positions.append(arr.max())
        left_most_positions.append(arr.min())
    
    right_most_positions = np.array(right_most_positions)
    left_most_positions = np.array(left_most_positions)
    # indeces_of_points_of_edges = np.argwhere(edges2==255)
    if right_most_positions.any() == False:
        min_right = 0
        max_right = 0
        min_left = 0
        max_left = 0
    else:
        min_right = right_most_positions.min()
        max_right = right_most_positions.max()
        min_left = left_most_positions.min()
        max_left = left_most_positions.max()

    an_img = Image.fromarray(img)
    
    an_img = an_img.convert("RGB")
    draw = ImageDraw.Draw(an_img) 
    draw.line((max_right, 0, max_right, img.shape[1]), fill=(0, 255, 0), width=1)
    draw.line((min_right, 0, min_right, img.shape[1]), fill=(0, 255, 0), width=1)
    draw.line((max_left, 0, max_left, img.shape[1]), fill=(255, 0, 0), width=1)
    draw.line((min_left, 0, min_left, img.shape[1]), fill=(255, 0, 0), width=1)
    # an_img.show()img
    if img_save_path is not None:
        an_img.save(img_save_path)
    return min_left, max_left, min_right, max_right

