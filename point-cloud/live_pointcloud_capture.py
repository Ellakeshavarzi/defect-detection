import depthai as dai
import cv2
import numpy as np
import open3d as o3d
import time

# Create pipeline
pipeline = dai.Pipeline()

# Mono cameras (stereo pair)
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# RGB camera
colorCam = pipeline.createColorCamera()
colorCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)

# Stereo depth
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)  # Align with rectified left, not RGB
stereo.setSubpixel(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)
stereo.initialConfig.setConfidenceThreshold(200)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# Link mono to stereo
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Output streams
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
colorCam.video.link(xoutRgb.input)

# Boot device and enable IR flood light
device = dai.Device(pipeline)
device.setIrFloodLightBrightness(1000)  # mA; range 0–1500

with device:
    depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False)
    rgbQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False)

    print("📡 Live view started. Press 's' to save point cloud. Press 'q' to quit.")

    while True:
        depthFrame = depthQueue.get().getFrame()
        rgbFrame = rgbQueue.get().getCvFrame()

        # Debug depth values
        print("Depth range:", np.min(depthFrame), "-", np.max(depthFrame))

        # Display depth
        depthVis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depthVis = cv2.applyColorMap(depthVis.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("RGB", rgbFrame)
        cv2.imshow("Depth", depthVis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("❌ Quit.")
            break
        elif key == ord('s'):
            print("💾 Saving point cloud and screenshots...")
            timestamp = int(time.time())
            cv2.imwrite(f"rgb_{timestamp}.png", rgbFrame)
            cv2.imwrite(f"depth_{timestamp}.png", depthVis)

            height, width = depthFrame.shape
            fx = fy = 445.0
            cx, cy = width // 2, height // 2

            points = []
            colors = []
            for v in range(height):
                for u in range(width):
                    Z = depthFrame[v, u] / 1000.0
                    if Z == 0: continue
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append([X, Y, Z])
                    colors.append(rgbFrame[v, u] / 255.0)

            if points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

                filename = f"pointcloud_{timestamp}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                print(f"✅ Saved: {filename}")
                o3d.visualization.draw_geometries([pcd])
            else:
                print("⚠️ No valid 3D points found. Try again.")
