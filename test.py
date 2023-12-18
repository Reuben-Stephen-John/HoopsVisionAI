import os
from ultralytics import YOLO
import cv2

mdl=YOLO(r'runs\detect\train4\weights\best.pt')
videofile='video_2023-12-18_14-54-11'
r=mdl.predict(source="static/input/"+videofile+'.mp4', project="static/output/",name="pred_"+videofile,save=True)

cap=cv2.VideoCapture("static/output/"+"pred_"+videofile+f"/{videofile}.avi")
if (cap.isOpened()== False):
    print("Error opening video file")
while (cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2

# # Load the pretrained YOLO model
# model = YOLO(r'runs/detect/train/weights/best.pt')
# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture a frame from the webcam
#     ret, frame = cap.read()

#     # Perform object detection on the frame
#     results = model(frame)

#     # Iterate through the list of results for each frame
#     for result in results.xyxy[0]:
#         # Extract information about the detected object
#         label = int(result[5])  # class label
#         confidence = float(result[4])  # confidence score
#         box = result[:4].int().cpu().numpy()  # bounding box coordinates

#         # Draw bounding box on the frame
#         color = (0, 255, 0)  # green color
#         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

#         # Display class label and confidence
#         label_text = f"{model.names[label]}: {confidence:.2f}"
#         cv2.putText(frame, label_text, (box[0], box[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Show the frame
#     cv2.imshow('YOLO Object Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()

