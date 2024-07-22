import numpy as np
import cv2

bg1_image = cv2.imread('Module 2 - Calculus and Linear Algegra, and theirs Applications to AI/Week 5/images/GreenBackground.png', 0)
bg1_image = cv2.resize(bg1_image, (678, 381))

ob_image = cv2.imread('Module 2 - Calculus and Linear Algegra, and theirs Applications to AI/Week 5/images/Object.png', 0)
ob_image = cv2.resize(ob_image, (678, 381))

bg2_image = cv2.imread('Module 2 - Calculus and Linear Algegra, and theirs Applications to AI/Week 5/images/NewBackground.jpg', 0)
bg2_image = cv2.resize(bg2_image, (678, 381))

def compute_difference(bg_img, input_img):
    difference_single_channel = cv2.absdiff(bg_img, input_img)
    return difference_single_channel

difference_single_channel = compute_difference(bg1_image, ob_image)
cv2.imshow('', difference_single_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

def compute_binary_mask(difference_single_channel):
    threshold = 50
    _, difference_binary = cv2.threshold(difference_single_channel, threshold, 255, cv2.THRESH_BINARY)
    return difference_binary

binary_mask = compute_binary_mask(difference_single_channel)
cv2.imshow('', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

def replace_background(bg1_image, bg2_image, ob_image):
    binary_mask = compute_binary_mask(compute_difference(bg1_image, ob_image))
    bg1_image = cv2.resize(bg1_image, (678, 381))
    bg2_image = cv2.resize(bg2_image, (678, 381))
    ob_image = cv2.resize(ob_image, (678, 381))
    bg1_image = cv2.bitwise_and(bg1_image, bg1_image, mask = cv2.bitwise_not(binary_mask))
    bg2_image = cv2.bitwise_and(bg2_image, bg2_image, mask = binary_mask)
    output = cv2.add(bg1_image, bg2_image)
    output = cv2.add(output, ob_image)
    return output

output = replace_background(bg1_image, bg2_image, ob_image)

cv2.imshow("Output Image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
