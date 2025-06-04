import cv2
import depthai as dai
import numpy as np


def calculate_layer_height(depth_map):
    # Convert to grayscale if not already
    if len(depth_map.shape) == 3:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Edge detection to find layer boundaries
    edges = cv2.Canny(depth_map, 50, 150)

    # Find contours (edges of each layer)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by y-coordinate (height)
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    # Calculate distance between consecutive contours (layer height)
    heights = []
    for i in range(len(contours) - 1):
        y1 = cv2.boundingRect(contours[i])[1]
        y2 = cv2.boundingRect(contours[i + 1])[1]
        height = abs(y2 - y1)
        heights.append(height)

    # Calculate average layer height
    avg_height = np.mean(heights) if heights else 0
    print(f"Estimated Layer Height: {avg_height:.2f} pixels")

    return avg_height


# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Output streams
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")

# Configure RGB camera
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Configure Mono cameras for depth
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Stereo depth configuration
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setExtendedDisparity(True)

# Linking
camRgb.video.link(xoutRgb.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # Get frames from queues
        inRgb = rgbQueue.get()
        inDepth = depthQueue.get()

        # Convert to OpenCV format
        frameRgb = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        # Normalize depth map for visualization
        depthFrame = (depthFrame * (255 / depthFrame.max())).astype(np.uint8)
        depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        # Display RGB and Depth frames
        cv2.imshow("RGB", frameRgb)
        layer_height = calculate_layer_height(depthFrame)
        print(f"Layer Height: {layer_height:.2f} pixels")


        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

