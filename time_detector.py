import cv2
from data import *


def angle_time(time):
    angle_step = 360 // 12
    if time != 0 and time < 6:
        mn = 15 + (time - 1) * angle_step
        mx = 15 + time * angle_step
    elif time != 0 and time > 6:
        mx = 360 - (15 + (time - 1) * angle_step)
        mn = 360 - (15 + time * angle_step)
    elif time == 6:
        mn = 165
        mx = 180
    elif time == 0:
        mn = 0
        mx = 15

    return mn, mx


def hand_control(hour, minute):
    message = ':-('
    if 1 <= minute <= 5:
        message = 'hour --- right hand, minute --- left'
        h = 'r'
        m = 'l'
    elif 7 <= minute <= 11:
        message = 'hour --- left hand, minute --- right'
        h = 'l'
        m = 'r'
    else:
        if 0 <= hour <= 6:
            message = 'hour --- left hand, minute --- right'
            h = 'l'
            m = 'r'
        elif 6 < hour <= 11:
            message = 'hour --- right hand, minute --- left'
            h = 'r'
            m = 'l'
    return message, h, m


def arrow(image, x_med, y_med, x_r_hand, y_r_hand, x_l_hand, y_l_hand, hour):
    w, h, _ = image.shape
    x0 = int(x_med * h)
    y0 = int(y_med * w)
    xr = int(x_r_hand * h)
    yr = int(y_r_hand * w)
    xl = int(x_l_hand * h)
    yl = int(y_l_hand * w)
    clr_h = (255, 170, 66)    # blue
    clr_m = (232, 207, 200)   # gray 
    cv2.putText(image, 'HOUR', (1060, 40),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, clr_h, 1, cv2.LINE_AA)
    cv2.putText(image, 'MINUTE', (1060, 80),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, clr_m, 1, cv2.LINE_AA)
    if hour == 'r':
        cv2.line(image, (x0, y0), (xr, yr), clr_h, 25)
        cv2.line(image, (x0, y0), (xl, yl), clr_m, 25)
    else:
        cv2.line(image, (x0, y0), (xr, yr), clr_m, 25)
        cv2.line(image, (x0, y0), (xl, yl), clr_h, 25)
    cv2.circle(image, (x0, y0), 60, (255, 170, 66), -1)


def detect(image, loc_hour, loc_minute, x_med, y_med, x_nose, y_nose, x_r_hand, y_r_hand, x_l_hand, y_l_hand, game_mode=True, draw_arrow=True):
    message, h, m = hand_control(loc_hour, loc_minute)
    h_angle_min, h_angle_max = angle_time(loc_hour)
    m_angle_min, m_angle_max = angle_time(loc_minute)

    if draw_arrow:
        arrow(image, x_med, y_med, x_r_hand, y_r_hand, x_l_hand, y_l_hand, h)

    angle_left_hand = calculate_angle([x_nose, y_nose], [x_med, y_med], [x_l_hand, y_l_hand])
    angle_right_hand = calculate_angle([x_nose, y_nose], [x_med, y_med], [x_r_hand, y_r_hand])

    flag = 0
    if (1 <= loc_minute <= 5) and (x_l_hand >= x_nose):  # m = 'l'
        if ((1 <= loc_hour <= 5) and (x_r_hand >= x_nose)) or ((7 <= loc_hour <= 11) and (x_r_hand <= x_nose)):
            flag = 1
        elif loc_hour == 0 or loc_hour == 6:
            flag = 1
    elif (7 <= loc_minute <= 11) and (x_r_hand <= x_nose):  # m = 'r'
        if ((1 <= loc_hour <= 5) and (x_l_hand >= x_nose)) or ((7 <= loc_hour <= 11) and (x_l_hand <= x_nose)):
            flag = 1
        elif loc_hour == 0 or loc_hour == 6:
            flag = 1
    elif loc_minute == 0 or loc_minute == 6:
        if ((1 <= loc_hour <= 5) and (x_l_hand >= x_nose)) or ((7 <= loc_hour <= 11) and (x_r_hand <= x_nose)):
            flag = 1
        elif loc_hour == 0 or loc_hour == 6:
            flag = 1

    if flag == 1:
        if h == 'r' and m == 'l':
            done = (m_angle_min <= angle_left_hand <= m_angle_max) and (h_angle_min <= angle_right_hand <= h_angle_max)
        else:
            done = (m_angle_min <= angle_right_hand <= m_angle_max) and (h_angle_min <= angle_left_hand <= h_angle_max)
    return done