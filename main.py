import cv2

# Load the pre-trained hand cascade classifier
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')

# Function to detect hands in the frame
def detect_hands(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Return the detected hands
    return hands

# Main function to capture video from webcam and detect hands
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame is captured successfully
        if not ret:
            print("Failed to capture frame")
            break
        
        # Detect hands in the frame
        hands = detect_hands(frame)
        
        # Draw rectangles around the detected hands
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        
        # Display the frame with detected hands
        cv2.imshow('Hand Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
