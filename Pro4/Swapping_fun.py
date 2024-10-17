import dlib
import numpy as np
import cv2
import os

def swap_faces(source_image, target_image):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the shape predictor file
    predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces in both images
    source_faces = detector(source_image)
    target_faces = detector(target_image)

    if len(source_faces) == 0 or len(target_faces) == 0:
        raise ValueError("No faces detected in one or both images")

    # Get the first face from each image
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Get facial landmarks
    source_landmarks = predictor(source_image, source_face)
    target_landmarks = predictor(target_image, target_face)

    # Convert landmarks to NumPy arrays
    source_points = np.array([(p.x, p.y) for p in source_landmarks.parts()])
    target_points = np.array([(p.x, p.y) for p in target_landmarks.parts()])

    # Calculate the convex hull of the facial landmarks
    hull = cv2.convexHull(target_points)

    # Create a mask for the face
    mask = np.zeros(target_image.shape[:2], dtype=np.float64)
    cv2.fillConvexPoly(mask, hull, 1)

    # Expand mask to 3 channels
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Calculate the affine transform to align the source face with the target face
    M = cv2.estimateAffinePartial2D(source_points, target_points)[0]

    # Warp the source image to align with the target image
    warped_source = cv2.warpAffine(source_image, M, (target_image.shape[1], target_image.shape[0]), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)

    # Combine the warped source image with the target image using the mask
    combined = (warped_source * mask + target_image * (1 - mask)).astype(np.uint8)

    # Blend the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    blur_amount = 5
    combined = cv2.blur(combined, (blur_amount, blur_amount))

    # Final combination
    result = (combined * mask + target_image * (1 - mask)).astype(np.uint8)

    return result