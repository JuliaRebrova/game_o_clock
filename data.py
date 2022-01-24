import numpy as np
import random as rd
import cv2
import os


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def random_time():
    hour = rd.randint(0, 11)
    minute = rd.randint(0, 11)
    while minute == hour:
        minute = rd.randint(0, 11)
    return hour, minute


def pose_correct(image, text, angle, correct_max, y, correct_min=0):
    if correct_min <= angle <= correct_max:
        cv2.putText(image, text, (10, y),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        return 0, image
    else:
        return 1, image


def time_initialization(round_len=2):
    round_time = []
    for i in range(round_len + 1):
        round_time.append(random_time())
    return round_time


def segmentation(image, results_segmentation_mask, bg_img_path='imgs/cows.jpg', bg_img=None, im=0):
    window_width, window_height, _ = image.shape
    if not im:
        bg_img = cv2.imread(os.path.join(bg_img_path), cv2.IMREAD_UNCHANGED)
    bg_img = cv2.resize(bg_img, (window_height, window_width))
    _, _, can = bg_img.shape
    if can > 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2BGR)
    condition = np.stack((results_segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = bg_img
    image = np.where(condition, image, bg_image)
    return image


def ready_strt(delta, image, d_time=3.0):
    x, y, _ = image.shape
    y, x = int(0.5 * x), int(0.5 * y)
    if 0.0 <= delta <= d_time:
        cv2.circle(image, (x, y), 100, (232, 207, 200), -1)  # gray 
        cv2.putText(image, 'Ready!', (x - 50, y),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return False, image
    elif d_time < delta <= 2 * d_time:
        cv2.circle(image, (x, y), 100, (255, 170, 66), -1)   # blue
        cv2.putText(image, 'Steady!', (x - 55, y),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return False, image
    elif 2 * d_time < delta <= 2.1 * d_time:
        cv2.circle(image, (x, y), 100, (175, 255, 46), -1) # green
        cv2.putText(image, 'GO!', (x - 23, y),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return False, image
    else:
        return True, image


