import cv2
import depthai as dai
import time
import os
import numpy as np


# Directory to store images
save_path = "./brick_images"
os.makedirs(save_path, exist_ok=True)

# Capture interval (in seconds)
interval = 10  # Change this value as needed

# Create pipeline
pipeline = dai.Pipeline()

# Set up mono (grayscale) cameras for depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Optional depth settings
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# XLink output for depth
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

# Connect to device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    print("Depth camera is running... Press 'q' to quit.")

    last_capture_time = time.time()

    while True:
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()

        # Normalize depth map to 8-bit image for display and saving
        depthFrameNorm = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depthFrameColor = cv2.applyColorMap(depthFrameNorm.astype(np.uint8), cv2.COLORMAP_JET)

        # Check if it's time to capture
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            depth_filename = f"{save_path}/depth_{timestamp}.png"

            # Save the colorized depth image
            cv2.imwrite(depth_filename, depthFrameColor)
            print(f"Depth image saved: {depth_filename}")

            last_capture_time = current_time

        # Display
        cv2.imshow("Depth Map", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
