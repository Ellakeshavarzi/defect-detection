import depthai as dai
import cv2
import numpy as np
import open3d as o3d

# Create pipeline
pipeline = dai.Pipeline()

# Define sources
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Updated
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

left = pipeline.createMonoCamera()
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Updated
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

right = pipeline.createMonoCamera()
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Updated
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Stereo depth node
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(200)  # Updated
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Updated
left.out.link(stereo.left)
right.out.link(stereo.right)

# Output streams
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Start device
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=True)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

    depthFrame = depthQueue.get().getFrame()
    rgbFrame = rgbQueue.get().getCvFrame()

    # Debug: Check depth range
    print("Depth min:", np.min(depthFrame), "max:", np.max(depthFrame))
    if np.max(depthFrame) == 0:
        print("⚠️ Warning: Depth data is empty. Try placing textured objects in view.")
        exit()

    # Show depth for visual confirmation
    depthVis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
    depthVis = cv2.applyColorMap(depthVis.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow("Depth", depthVis)
    cv2.imshow("RGB", rgbFrame)
    cv2.waitKey(1)

    # Camera intrinsics (approximate, or use calibration if needed)
    height, width = depthFrame.shape
    fx = fy = 445.0
    cx, cy = width // 2, height // 2

    # Convert depth map to point cloud
    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            Z = depthFrame[v, u] / 1000.0  # mm to meters
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(rgbFrame[v, u] / 255.0)

    # Make sure we have points
    if not points:
        print("❌ No valid depth points found. Cannot generate point cloud.")
        exit()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Save and show
    o3d.io.write_point_cloud("oakd_pointcloud.ply", pcd)
    print("✅ Saved point cloud as 'oakd_pointcloud.ply'")
    o3d.visualization.draw_geometries([pcd])
