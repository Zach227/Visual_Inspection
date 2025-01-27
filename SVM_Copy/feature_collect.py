from pathlib import Path

from PIL import Image, ImageOps                                  # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm
import cv2
THRESHOLD_VALUE = 180
# open source implementation of LBP
from skimage.feature import local_binary_pattern
# data preprocessing module in scikit-learn
from sklearn import preprocessing
# SVM implementation in scikit-learn
from sklearn.svm import LinearSVC


plt.rcParams['font.size'] = 11

# LBP function params
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
n_bins = n_points + 2

def compute_ratio(w, h):
    if w == 0  or h == 0:
        return 1
    return min(w,h) / max(w,h)

def compute_lbp(arr):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    lbp = local_binary_pattern(arr, n_points, radius, METHOD)
    lbp = lbp.ravel()
    # feature_len = int(lbp.max() + 1)
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    # feature /= np.linalg.norm(feature, ord=1)
    return feature

def compute_area(arr):
# Read the image
    # edges = cv2.Canny(arr, 100, 200)

    # Find contours in the edges
    _, thresholded = cv2.threshold(arr, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        # Find contours in the edges

    chosen_frame = thresholded

    contours, _ = cv2.findContours(chosen_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize variables to track the largest contour
    largest_contour = None
    max_area = 0
    # Iterate over all contours to find the largest one
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > max_area:
            max_area = area
            largest_contour = cont  

    x, y, w, h = cv2.boundingRect(largest_contour)
    return max_area, (x, y, x+w, y+h)


def compute_circularity(arr):
    # Apply the Canny edge detection
    # edges = cv2.Canny(arr, 50, 200)
    _, thresholded = cv2.threshold(arr, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # Find contours in the edges
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to track the largest contour
    largest_contour = None
    max_area = 0

    # Iterate over all contours to find the largest one
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > max_area:
            max_area = area
            largest_contour = cont

    # If a contour was found
    if largest_contour is not None:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(largest_contour, True)

        # Avoid division by zero if perimeter is zero (possible in degenerate cases)
        if perimeter == 0:
            return 0

        # Calculate the circularity using the formula
        circularity = (4 * np.pi * max_area) / (perimeter ** 2)
        return circularity
    else:
        return 0  # No contour found


def load_data(tag='training-set'):
    """Load (training/test) data from the directory.
    Also do preprocessing to extra features. 
    """
    tag_dir = Path.cwd() / tag
    print(tag_dir)
    area_vec = []
    lbp_vec = []
    circle_vec = []
    ratio_vec = []
    vec = []
    cat = []
    mean_color_vec = []
    for cat_dir in tag_dir.iterdir():
        cat_label = cat_dir.stem
        print(cat_label)
        for img_path in cat_dir.glob('*.png'):
            img = Image.open(img_path.as_posix())
            #print(img_path.as_posix(), img.mode)
            if img.mode != 'L':
                img = ImageOps.grayscale(img)
                img.save(img_path.as_posix())
            arr = np.array(img)

            #Window the array
            x, y, w, h = 70, 0, 450, 400
            arr = arr[y:y+h, x:x+w]
            #Feature 1 See if Skittle is round
            # feature = compute_lbp(arr)
            # edges = cv2.Canny(arr, 100, 200)

            # _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
            # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # area = cv2.contourArea(contour)
            area, box = compute_area(arr)
            
            if box and any(box):  # Check if box exists and contains non-zero values
                skittle_box = arr[box[1]:box[3], box[0]:box[2]]
            else:
                continue
                # skittle_box = arr  # Or handle it differently based on your needs
            #Feature 2 Use Dims for stuff
            lbp = compute_lbp(skittle_box)
            mean_color = np.mean(skittle_box)  
            mean_color = np.array([mean_color])
            area = np.array([area])
            circle = compute_circularity(arr)
            ratio = compute_ratio(box[2] - box[0], box[3] - box[1])
            ratio = np.array([ratio])
            lbp /= np.linalg.norm(lbp, ord=1)
            # print(np.array([lbp, area]).shape)
            print(ratio)
            area_vec.append(area)
            lbp_vec.append(lbp)
            circle_vec.append(circle)
            ratio_vec.append(ratio)
            mean_color_vec.append(mean_color)
            cat.append(cat_label)


    print("-----------------------------------------------")
    area_array = np.array(area_vec)
    lbp_array = np.array(lbp_vec)
    circle_array = np.array(circle_vec)
    ratio_array = np.array(ratio_vec)
    mean_color_array = np.array(mean_color_vec)
    print(area_array.shape)
    print(lbp_array.shape)
    combined = np.hstack([lbp_array, area_array.reshape(-1,1)])
    combined = np.hstack([combined, circle_array.reshape(-1,1)])
    combined = np.hstack([combined, ratio_array.reshape(-1,1)])
    combined = np.hstack([combined, mean_color_array.reshape(-1,1)])


    print(combined.shape)
    # print(combined)
    print("-----------------------------------------------")

    return combined, cat


vec_train, cat_train = load_data('video_images_3')        # load training data
np.save("vec_train.npy", vec_train)
np.save("cat_train.npy", cat_train)