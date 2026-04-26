import cv2
import numpy as np
from mss import mss
from nudenet import NudeDetector

# 1. Initialize the AI
detector = NudeDetector()

# 2. Define the explicit classes you want to block
# Updated for NudeNet V3
EXPLICIT_CLASSES = [
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED"
]
# Helper function to heavily pixelate an image region
def pixelate_region(image, blocks=7):
    # 'blocks' determines how chunky the pixels are. Lower = more censored.
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    # Shrink down to a tiny grid
    small = cv2.resize(image, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    # Blow it back up without smoothing (INTER_NEAREST creates the blocks)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# 3. Define the screen area to monitor
monitor = {"top": 100, "left": 100, "width": 800, "height": 600} # Specific Window area

with mss() as sct:
    while True:
        # Grab the screen pixels
        screen_shot = sct.grab(monitor)
        
        # Convert raw pixels to a NumPy array for OpenCV
        frame = np.array(screen_shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run detection
        detections = detector.detect(frame)

        # Loop through AI results and apply the filter
        for det in detections:
            # Check confidence AND check if the detected class is in our explicit list
            if det['score'] > 0.25 and det['class'] in EXPLICIT_CLASSES:
                x, y, w, h = det['box']
                
                # Ensure coordinates are integers and within frame bounds
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Slicing the Region of Interest (ROI)
                roi = frame[y:y+h, x:x+w]
                
                if roi.size > 0:
                    # Apply the heavy pixelation effect instead of a soft blur
                    censored_roi = pixelate_region(roi, blocks=7)
                    
                    # Alternatively, if you just want an extreme black box, you could do:
                    # censored_roi = np.zeros_like(roi)
                    
                    frame[y:y+h, x:x+w] = censored_roi

        # Show the "Filtered" view
        cv2.imshow("Real-Time Filtered Stream", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()