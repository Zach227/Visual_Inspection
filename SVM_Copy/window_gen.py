import cv2
from pathlib import Path

# Define region of interest (ROI) dimensions
roi_x, roi_y, roi_w, roi_h = 70, 0, 450, 300

tag = "video_images_2"
tag_dir = Path.cwd() / tag
print(tag_dir)
vec = []
cat = []
for cat_dir in tag_dir.iterdir():
    for img_path in cat_dir.glob('*.png'):
 
        # Read the image
        frame = cv2.imread(img_path)
        # Extract the region of interest
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply edge detection

        # edges = cv2.Canny(gray, 50, 200)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
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
        # Draw a bounding box around the largest contour if found
        roi = thresholded
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 255, 255), 2)  # Green rectangle with thickness 2
        # Display the result
        cv2.putText(roi,  img_path.name, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)

        cv2.imshow("Largest Contour with Bounding Box", roi)
        # Wait for a key press and close the window
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cv2.destroyAllWindows()
