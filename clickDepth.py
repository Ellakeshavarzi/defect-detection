import cv2
import depthai as dai
import numpy as np

# Set up the pipeline
pipeline = dai.Pipeline()

# Set up mono cameras for depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Global variable to hold depth frame for click callback
depth_frame_mm = None

# Mouse callback function
def on_mouse_click(event, x, y, flags, param):
    global depth_frame_mm
    if event == cv2.EVENT_LBUTTONDOWN and depth_frame_mm is not None:
        depth = depth_frame_mm[y, x]
        print(f"Depth at ({x}, {y}) = {depth} mm")

# Start device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    cv2.namedWindow("Depth Map")
    cv2.setMouseCallback("Depth Map", on_mouse_click)

    print("Click on the depth map to get depth in mm. Press 'q' to quit.")

    while True:
        inDepth = depthQueue.get()
        depth_frame_mm = inDepth.getFrame()  # Raw depth map in mm

        # Normalize for visualization
        depthFrameNorm = cv2.normalize(depth_frame_mm, None, 0, 255, cv2.NORM_MINMAX)
        depthFrameColor = cv2.applyColorMap(depthFrameNorm.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("Depth Map", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
