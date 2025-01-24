import cv2
from pathlib import Path

from pathlib import Path

from PIL import Image, ImageOps                                  # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm
import cv2
import pickle
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
    edges = cv2.Canny(arr, 100, 200)
    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    edges = cv2.Canny(arr, 50, 200)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

def load_image(frame, tag='training-set'):
    """Load (training/test) data from the directory.
    Also do preprocessing to extra features. 
    """
    # tag_dir = Path.cwd() / tag
    # print(tag_dir)
    area_vec = []
    lbp_vec = []
    circle_vec = []
    # vec = []
    # cat = []
    # for cat_dir in tag_dir.iterdir():
        # cat_label = cat_dir.stem
        # print(cat_label)
        # for img_path in cat_dir.glob('*.png'):
    # img = Image.open(img_path.as_posix())
    # #print(img_path.as_posix(), img.mode)
    # if img.mode != 'L':
    #     img = ImageOps.grayscale(img)
    #     img.save(img_path.as_posix())
    arr = np.array(frame)
    #make the frame black and white



    # arr = np.array(img)

    #Window the array
    x, y, w, h = 140, 0, 535, 480
    roi = arr[y:y+h, x:x+w]
    

    # _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # area = cv2.contourArea(contour)
    area, box = compute_area(roi)
    skittle_box = roi[box[1]:box[3], box[0]:box[2]]

    # Feature 2 Use Dims for stuff
    lbp = compute_lbp(skittle_box)
    area = np.array([area])
    circle = compute_circularity(roi)
    lbp /= np.linalg.norm(lbp, ord=1)
    # print(np.array([lbp, area]).shape)
    # print(circle)
    area_vec.append(area)
    lbp_vec.append(lbp)
    circle_vec.append(circle)
    # cat.append(cat_label)

    # print("-----------------------------------------------")
    area_array = np.array(area_vec)
    lbp_array = np.array(lbp_vec)
    circle_array = np.array(circle_vec)
    # print(area_array.shape)
    # print(lbp_array.shape)
    combined = np.hstack([lbp_array, area_array.reshape(-1,1)])
    combined = np.hstack([combined, circle_array.reshape(-1,1)])

    # print(combined.shape)
    # print(combined)
    # print("-----------------------------------------------")
    return combined, box





# Define region of interest (ROI) dimensions
roi_x, roi_y, roi_w, roi_h = 150, 0, 525, 480

tag = "train_images"
tag_dir = Path.cwd() / tag
print(tag_dir)
vec = []
cat = []

video = cv2.VideoCapture("../sample_video.avi")

# Load the model
with open('linear_svc_model.pkl', 'rb') as file:
    loaded_clf = pickle.load(file)


while True:
    ret, frame = video.read()
    if not ret:
        print(f"Bad Frame {ret}")
        break
    roi_x, roi_y, roi_w, roi_h = 150, 0, 525, 270
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    combined, box = load_image(gray)

    pred = loaded_clf.predict(combined)

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    text = ""
    if pred[0] == 0:
        text = "Acceptable"
    elif pred[0] == 1:
        text ="Bad"
    elif pred[0] == 2:
        text = "Good"
    # if w-x * h-y > 150:
    if ((w-x) * (h-y)) > 300:
        cv2.rectangle(roi, (x+150, y), (w+150, h), (0, 255, 0), 2)  # Green rectangle with thickness 2
        cv2.putText(roi,  text, (x+150, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2)
        
    print(pred)
    cv2.imshow("asdf", roi)

    cv2.waitKey(40)
