import cv2
import depthai as dai
import time
import os

# Directory to store images
save_path = "./brick_images"
os.makedirs(save_path, exist_ok=True)

# Capture interval (in seconds)
interval = 10  # Change this value to adjust capture frequency

# Create pipeline
pipeline = dai.Pipeline()

# Setup Color Camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# XLinkOut for RGB output
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("Camera is running... Press 'q' to quit.")

    last_capture_time = time.time()

    while True:
        # Get RGB frame
        inRgb = rgbQueue.get()
        frame = inRgb.getCvFrame()

        # Check if enough time has passed since the last capture
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            # Generate a timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{save_path}/brick_{timestamp}.jpg"

            # Save the captured frame
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Image saved: {filename}")

            # Update the last capture time
            last_capture_time = current_time

        # Show the live camera feed
        cv2.imshow("Brick Image", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
