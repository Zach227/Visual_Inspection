from pathlib import Path

from PIL import Image, ImageOps                                  # for image I/O
 
from PIL import Image                                   # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm

import pickle
import cv2

# open source implementation of LBP
from skimage.feature import local_binary_pattern
# data preprocessing module in scikit-learn
from sklearn import preprocessing
# SVM implementation in scikit-learn
from sklearn.svm import LinearSVC

THRESHOLD_VALUE = 180


plt.rcParams['font.size'] = 11

# LBP function params
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
n_bins = n_points + 2

def compute_haralick_features(img):
    glcm = greycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    return np.array([contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0]])


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
    # print(tag_dir)
    area_vec = []
    lbp_vec = []
    circle_vec = []
    ratio_vec = []
    vec = []
    cat = []
    mean_color_vec = []
    for cat_dir in tag_dir.iterdir():
        cat_label = cat_dir.stem
        # print(cat_label)
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
            # print(ratio)
            area_vec.append(area)
            lbp_vec.append(lbp)
            circle_vec.append(circle)
            ratio_vec.append(ratio)
            mean_color_vec.append(mean_color)
            cat.append(cat_label)


    # print("-----------------------------------------------")
    area_array = np.array(area_vec)
    lbp_array = np.array(lbp_vec)
    circle_array = np.array(circle_vec)
    ratio_array = np.array(ratio_vec)
    mean_color_array = np.array(mean_color_vec)
    # print(area_array.shape)
    # print(lbp_array.shape)
    combined = np.hstack([lbp_array, area_array.reshape(-1,1)])
    combined = np.hstack([combined, circle_array.reshape(-1,1)])
    combined = np.hstack([combined, ratio_array.reshape(-1,1)])
    combined = np.hstack([combined, mean_color_array.reshape(-1,1)])


    # print(combined.shape)
    # print(combined)
    # print("-----------------------------------------------")

    return combined, cat


vec_train = np.load("vec_train.npy")
cat_train = np.load("cat_train.npy")
le = preprocessing.LabelEncoder()
le.fit(cat_train)
label_train = le.transform(cat_train)

vec_test, cat_test = load_data('video_images_3')              # load test data
label_test = le.transform(cat_test)

def get_conf_mat(y_pred, y_target, n_cats):
    """Build confusion matrix from scratch.
    (This part could be a good student assignment.)
    """
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm


def vis_conf_mat(conf_mat, cat_names, acc):
    """Visualize the confusion matrix and save the figure to disk."""
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    # plt.show()
    _filename = 'train_confusion_mat.png'
    plt.savefig(_filename, bbox_inches='tight')


# SVM
clf = LinearSVC(random_state=0, tol=1e-5, max_iter=15000000, class_weight='balanced')
clf.fit(vec_train, label_train)             # fit SVM using training data

with open('linear_svc_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# evaluation
# prediction = clf.predict(vec_test)          # make prediction on the test data
prediction = clf.predict(vec_train)          # make prediction on the test data

# visualization
cmat = get_conf_mat(y_pred=prediction, y_target=label_test, n_cats=len(le.classes_))
cmat = get_conf_mat(y_pred=prediction, y_target=label_train, n_cats=len(le.classes_))

acc = cmat.trace() / cmat.shape[0]
vis_conf_mat(cmat, le.classes_, acc)