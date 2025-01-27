import cv2
from pathlib import Path

# Define region of interest (ROI) dimensions
roi_x, roi_y, roi_w, roi_h = 150, 0, 525, 480

tag = "train_images"
tag_dir = Path.cwd() / tag
print(tag_dir)
vec = []
cat = []

video = cv2.VideoCapture("../sample_video.avi")
i = 0
while True:  
 
        # Read the image
        ret, frame = video.read()
        frame_copy = frame.copy()
        if not ret:
            print(f"Bad Frame {ret}")
            break
        # Extract the region of interest
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply edge detection
        edges = cv2.Canny(gray, 100, 200)
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
        # Draw a bounding box around the largest contour if found
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle with thickness 2
        # Display the result
        cv2.imshow("Largest Contour with Bounding Box", frame)
        # Wait for a key press and close the window
        while True:
            key = cv2.waitKey(10) & 0xFF  # Wait slightly longer to ensure key capture

            if key == ord('q'):  # Quit the program
                break
            elif key == ord('e'):  # Save to general folder
                cv2.imwrite(f"./video_images_3/img{i}.png", frame_copy)
                i += 1
                break  # Exit the while loop and proceed to the next frame
            elif key == ord('g'):  # Save to 'Good' folder
                cv2.imwrite(f"./video_images_3/Good/img{i}.png", frame_copy)
                i += 1
                break  # Exit the while loop and proceed to the next frame
            elif key == ord('u'):  # Save to 'Acceptable' folder
                cv2.imwrite(f"./video_images_3/Acceptable/img{i}.png", frame_copy)
                i += 1
                break  # Exit the while loop and proceed to the next frame
            elif key == ord('b'):  # Save to 'Bad' folder
                cv2.imwrite(f"./video_images_3/Bad/img{i}.png", frame_copy)
                i += 1
                break  # Exit the while loop and proceed to the next frame

        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #     elif cv2.waitKey(1) & 0xFF == ord('e'):
        #         cv2.imwrite(f"./video_images_2/img{i}.png", frame_copy)
        #         i += 1
        #         break
        #     elif cv2.waitKey(1) & 0xFF == ord('g'):
        #         cv2.imwrite(f"./video_images_2/Good/img{i}.png", frame_copy)
        #         i += 1
        #         break
        #     elif cv2.waitKey(1) & 0xFF == ord('u'):
        #         cv2.imwrite(f"./video_images_2/Acceptable/img{i}.png", frame_copy)
        #         i += 1
        #         break
        #     elif cv2.waitKey(1) & 0xFF == ord('b'):
        #         cv2.imwrite(f"./video_images_2/Bad/img{i}.png", frame_copy)
        #         i += 1
        #         break

cv2.destroyAllWindows()
