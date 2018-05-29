# head_pose

Use opencv-python API, solvePnP method to estimate head pose.

Dlib is not must, only for face landmarks detection for this repository,
you can definitely change it to another face landmarks detection library, such as MTCNN.

# Usage

See test samples of test_head_pose_estimator.py

## To run test, put the test images in sample_images directory
```
python test_head_pose_estimator.py
```


# TO DO
Will add CNN to do head pose estimation, for many situations cannot get all the accurate needed points for PnP algorithm