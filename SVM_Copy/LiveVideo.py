import cv2
from pathlib import Path

from pathlib import Path

THRESHOLD_VALUE = 180


from collections import Counter
from PIL import Image, ImageOps                                  # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm
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


def load_image(frame, tag='training-set'):
    """Load (training/test) data from the directory.
    Also do preprocessing to extra features. 
    """
    # tag_dir = Path.cwd() / tag
    # print(tag_dir)
    area_vec = []
    lbp_vec = []
    circle_vec = []
    ratio_vec = []
    mean_color_vec = []

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
    x, y, w, h = 70, 0, 450, 400
    roi = arr[y:y+h, x:x+w]
    

    # _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # area = cv2.contourArea(contour)
    area, box = compute_area(roi)
    if box and any(box):  # Check if box exists and contains non-zero values
        skittle_box = arr[box[1]:box[3], box[0]:box[2]]
    else:
        return None, None
        # skittle_box = arr  # Or handle it differently based on your needs

    # Feature 2 Use Dims for stuff
    lbp = compute_lbp(skittle_box)
    area = np.array([area])
    circle = compute_circularity(roi)
    mean_color = np.mean(skittle_box)  
    ratio = compute_ratio(box[2] - box[0], box[3] - box[1])

    mean_color = np.array([mean_color])
    ratio = compute_ratio(box[2] - box[0], box[3] - box[1])


    lbp /= np.linalg.norm(lbp, ord=1)
    # print(np.array([lbp, area]).shape)
    # print(circle)
    area_vec.append(area)
    lbp_vec.append(lbp)
    circle_vec.append(circle)
    ratio_vec.append(ratio)
    mean_color_vec.append(mean_color)

    # cat.append(cat_label)

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
    return combined, box





# Define region of interest (ROI) dimensions

tag = "train_images"
tag_dir = Path.cwd() / tag
print(tag_dir)
vec = []
cat = []

video = cv2.VideoCapture("../sample_video.avi")

# Load the model
with open('linear_svc_model.pkl', 'rb') as file:
    loaded_clf = pickle.load(file)
folder_paths = []
folder_path = "./video_images_2/Good"
folder_path = Path(folder_path)

folder_paths.append(folder_path)

folder_path = "./video_images_2/Bad"
folder_path = Path(folder_path)

folder_paths.append(folder_path)

folder_path = "./video_images_2/Acceptable"
folder_path = Path(folder_path)

folder_paths.append(folder_path)
for folder in folder_paths:
    break
    for image_path in folder.glob('*.png'):
        # Open the image using PIL
        image = Image.open(image_path)
        
        # Convert the image to a NumPy array (you can use OpenCV here as well)
        frame = np.array(image)
        
        # Alternatively, if you prefer using OpenCV to read images as frames:
        # frame = cv2.imread(str(image_path))
        
        # Here you can process the frame as needed, for example:
        # - Apply filters
        # - Convert to grayscale
        # - Display the frame in OpenCV
        roi_x, roi_y, roi_w, roi_h = 70, 0, 450, 400
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        combined, box = load_image(frame)
        if box and any(box) and combined is not None:  # Check if box exists and contains non-zero values
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
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)  # Green rectangle with thickness 2
                cv2.putText(frame,  f"{text, pred[0]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2)

            print(pred)
        # if pred[0] == 2:
        cv2.imshow("asdf", frame)


        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

current_guess = 0

check_point_1 = False
check_point_2 = False
check_point_3 = False

guesses=[]
y_test_fails = 0
last_y = 400
last_last_y = 400

frames_without_box = 0

camera = cv.VideoCapture(0)

bash_script = "./camera_settings.sh"
result = subprocess.run(
    ["bash", bash_script],
    text=True,  # To handle output as strings
    capture_output=True,  # Captures stdout and stderr
    check=True  # Raises exception if the command fails
)

while True:
    ret, frame = camera.read()
    if not ret:
        print(f"Bad Frame {ret}")
        break
    roi_x, roi_y, roi_w, roi_h = 70, 0, 450, 400
    # roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(frame)  # Convert NumPy array to PIL Image
    img = ImageOps.grayscale(img)
    combined, box = load_image(img)
    if box and any(box) and combined is not None and box[0] > 50: # Check if box exists and contains non-zero values
        frames_without_box = 0
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
        if ((w-x) * (h-y)) > 200:
            cv2.rectangle(frame, (x+roi_x, y), (w+roi_x, h), (0, 255, 0), 2)  # Green rectangle with thickness 2
            cv2.putText(frame,  text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_w, roi_h), (255, 0, 0), 2)  # Green rectangle with thickness 2

        # print(pred)
        guesses.append(pred[0])
    else:
        frames_without_box += 1
    if box is None:
        y = 0
    else:
        y = box[1]
    
    # if (frames_without_box  == 3 or y_test_fails == 2) and len(guesses) > 20  :
    if check_point_1 and check_point_2 and check_point_3:
        counter = Counter(guesses)
        guess = ""
        most_common_guess = max(counter, key=counter.get)  # Find the key with the maximum count
        guess = ""
        if most_common_guess == 2 and counter[2] > 10:
            guess = "Good"
        elif most_common_guess == 1 and counter[1] > 10:
            guess = "Bad"
        else:
            guess = "Acceptable"        
        if len(guesses) > 15:
            print(f"THE CURRENT GUESS IS {guess} {counter}")
        guesses = []
        check_point_1 = False
        check_point_2 = False
        check_point_3 = False

    # if y * 1.1 < last_y and y * 1.1 < last_last_y:
    #     y_test_fails += 1
    #     if y_test_fails  == 2:
    #         print("Double Y Fails")
    if box is not None:
        y = box[1]
        x = box[0]
    else:
        x = 0
        y = 500
    # print(y, check_point_1, check_point_2, check_point_3)
    if y < 200 and y >= 50 and check_point_2 == False and check_point_3 == False and x > 15:
        check_point_1 = True
    elif y  < 300 and y >=200 and check_point_1 == True and check_point_3 == False:
        check_point_2 = True
    elif y < 375 and y >= 300 and check_point_1 == True and check_point_2 == True:
        check_point_3 = True
    else:
        check_point_1 = False
        check_point_2 = False
        check_point_3 = False
        guesses= [] 

    if check_point_1 == True and check_point_2 == True and check_point_3 == True:
        print("PASSED ALL CHECKS")
    cv2.imshow("asdf", frame)
    # print(check_point_1, check_point_2, check_point_3, box)
    cv2.waitKey(15)

    
