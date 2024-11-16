import cv2
import numpy as np

N = 7
confThresh = 0.4
inputW, inputH = 300, 300
mean = (104.0, 177.0, 123.0)

prototxt_path = "/mnt/d/py/deploy.prototxt"
model_path = "/mnt/d/py/res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def interpolate(frames, start_coords, end_coords):
    if start_coords is None or end_coords is None:
        return

    xLeft_start, yLeft_start, xRight_start, yRight_start = start_coords
    xLeft_end, yLeft_end, xRight_end, yRight_end = end_coords

    for i in range(1, N):
        alpha = i / N
        xLeft = int((1 - alpha) * xLeft_start + alpha * xLeft_end)
        yLeft = int((1 - alpha) * yLeft_start + alpha * yLeft_end)
        xRight = int((1 - alpha) * xRight_start + alpha * xRight_end)
        yRight = int((1 - alpha) * yRight_start + alpha * yRight_end)

        cv2.rectangle(frames[i], (xLeft, yLeft), (xRight, yRight), (0, 255, 0), 2)

def detect_face(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        frames.append(img)
        frame_count += 1

        if frame_count == N + 1:
            first_frame = frames[-(N + 1)]
            last_frame = img
            
            blob_first = cv2.dnn.blobFromImage(first_frame, 1.0, (inputW, inputH), mean, swapRB=False, crop=False) 
            net.setInput(blob_first) 
            detections_first = net.forward() 
            imgH_first, imgW_first = first_frame.shape[:2]
            face_coords_1 = None
            
            for i in range(detections_first.shape[2]): 
                confidence = detections_first[0, 0, i, 2]
                if confidence > confThresh:
                    xLeft_first = int(detections_first[0, 0, i, 3] * imgW_first)
                    yLeft_first = int(detections_first[0, 0, i, 4] * imgH_first)
                    xRight_first = int(detections_first[0, 0, i, 5] * imgW_first)
                    yRight_first = int(detections_first[0, 0, i, 6] * imgH_first)

                    face_coords_1 = (xLeft_first, yLeft_first, xRight_first, yRight_first)
                    cv2.rectangle(first_frame,
                                  (xLeft_first,yLeft_first),
                                  (xRight_first,yRight_first),
                                  (0 ,0, 255), 2)
                    break
            
            blob_last = cv2.dnn.blobFromImage(last_frame, 1.0, (inputW, inputH), mean, swapRB=False, crop=False) 
            net.setInput(blob_last) 
            detections_last = net.forward() 
            imgH_last, imgW_last = last_frame.shape[:2]
            face_coords_2 = None
            
            for i in range(detections_last.shape[2]): 
                confidence = detections_last[0, 0, i, 2]
                if confidence > confThresh:
                    xLeft_last = int(detections_last[0, 0, i, 3] * imgW_last)
                    yLeft_last = int(detections_last[0, 0, i, 4] * imgH_last)
                    xRight_last = int(detections_last[0, 0, i, 5] * imgW_last)
                    yRight_last = int(detections_last[0, 0, i, 6] * imgH_last)

                    face_coords_2 = (xLeft_last,yLeft_last,xRight_last,yRight_last)
                    cv2.rectangle(last_frame,
                                  (xLeft_last,yLeft_last),
                                  (xRight_last,yRight_last),
                                  (0 ,0 ,255), 2)
                    break

            interpolate(frames, face_coords_1 if face_coords_1 else None,
                        face_coords_2 if face_coords_2 else None)

            for frame in frames[-(N + 1):]:
                out.write(frame)

            frames.clear()
            frame_count = 0

    cap.release()
    out.release()

video_path = "/mnt/d/py/Gosling.mp4"
output_path = "/mnt/d/py/Gosling_processed.mp4"

detect_face(video_path ,output_path)