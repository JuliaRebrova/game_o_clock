import cv2
import mediapipe as mp
import numpy as np
import random as rd
from time_detector import detect
from data import *
from button import Button
import time
import pyglet

# Full screen
window_name = 'Game'

# video settings
cap = cv2.VideoCapture(0)
window_width = 1280
window_height = 720
cap.set(3, window_width)
cap.set(4, window_height)

# Interface and modes
menu = True
choose_selection = False
game_mode = False
easy_game = True
alarm_mode = False
hand_correction = False  # Advise to straighten arms or not
posture_correction = False  # Advise to straighten body or not
enable_segmentation = True  # Turn a segmentation on/off
draw_arrow = True # Draw arrows over the arms or not
landmarks_show = False # Draw pose landmarks or not

# Alarm mode settings
hour_alarm = 8
minute_alarm = 0 # from 0 till 12 (нoutput is multiplied by 5)
alarm_song = pyglet.media.Player()
source = pyglet.media.StreamingSource()
MediaLoad = pyglet.media.load("alarm_song.mp3")
alarm_song.queue(MediaLoad)

# buttons hold time (frames)
button_pause = 8
start_button_time = 0
exit_game_button_time = 0
exit_button_time = 0
hard_button_time = 0
easy_button_time = 0
alarm_button_time = 0
choose_exit_button_time = 0
button_song = pyglet.resource.media("song.wav", streaming=False)
sound_win = pyglet.resource.media("win.wav", streaming=False)

# Timers
d_time = 3.0 # Time for ready/steady/go
const_hard_game_round_time = 10.0 # seconds
hard_game_round_time = const_hard_game_round_time

# Time initialization for round
time_step = 10
time_index = 0
time_index_alarm = 0
round_len = 2
round_index = 0
round_done = 0

if game_mode:
    round_time = time_initialization()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                  enable_segmentation=enable_segmentation) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)   # flip image

        # MediaPipe color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # MediaPipe color
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # left elbow
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # right elbow
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # median angle
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            median = [(right_shoulder[0] + left_shoulder[0]) * 0.5, (right_shoulder[1] + left_shoulder[1]) * 0.5]

            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]

            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]

            palm_l_x = 0.5 * (left_pinky[0] + left_index[0]) * window_width
            palm_l_y = 0.5 * (left_pinky[1] + left_index[1]) * window_height
            palm_r_x = 0.5 * (right_pinky[0] + right_index[0]) * window_width
            palm_r_y = 0.5 * (right_pinky[1] + right_index[1]) * window_height

            angle_median = calculate_angle(nose, median, right_wrist)
            angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # MENU START
            if menu:
                image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/no_start_no_exit.jpg')
                start_button = Button([23, 299], 'Choose selection', [320, 74], (0, 0, 0))
                exit = Button([945, 299], 'Exit', [320, 74], (0, 0, 0))

                if start_button.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    start_button_time += 1
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/start_no_exit.png')
                    if start_button_time == button_pause:
                        menu = False
                        choose_selection = True
                        button_song.play()
                else:
                    start_button_time = 0

                if exit.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/exit_no_start.png')
                    exit_button_time += 1
                    if exit_button_time == button_pause:
                        button_song.play()
                        cap.release()
                        cv2.destroyAllWindows()
                else:
                    exit_button_time = 0
            # MENU END

            # SELECTION START
            if choose_selection:
                image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/no_buttons.png')
                easy_button = Button([20, 195], 'Start easy game', [350, 74], (0, 0, 0))
                hard_button = Button([20, 303], 'Start hard game', [350, 74], (0, 0, 0))
                alarm = Button([20, 409], 'Alarm', [350, 74], (0, 0, 0))
                choose_exit = Button([20, 518], 'Exit', [350, 74], (0, 0, 0))

                if easy_button.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    easy_button_time += 1
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/easy.png')
                    if easy_button_time == button_pause:
                        button_song.play()
                        menu = False
                        choose_selection = False
                        game_mode = True
                        easy_game = True
                        ready_flag = 1
                        ready_start = time.time()
                        round_time = time_initialization()
                        round_index = 0
                        round_done = 0
                else:
                    easy_button_time = 0

                if hard_button.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/hard.png')
                    hard_button_time += 1
                    if hard_button_time == button_pause:
                        button_song.play()
                        menu = False
                        choose_selection = False
                        game_mode = True
                        easy_game = False
                        ready_flag = 1
                        ready_start = time.time()
                        round_time = time_initialization()
                        round_index = 0
                        round_done = 0
                        hard_game_round_time = const_hard_game_round_time

                else:
                    hard_button_time = 0

                if alarm.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/alarm.png')
                    alarm_button_time += 1
                    if alarm_button_time == button_pause:
                        button_song.play()
                        alarm_mode = True
                        menu = False
                        choose_selection = False
                        alarm_flag = 1
                        time_index_alarm = 0
                        alarm_song = pyglet.media.Player()
                        source = pyglet.media.StreamingSource()
                        MediaLoad = pyglet.media.load("alarm_song.mp3")
                        alarm_song.queue(MediaLoad)
                else:
                    alarm_button_time = 0

                if choose_exit.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/exit.png')
                    choose_exit_button_time += 1
                    if choose_exit_button_time == button_pause:
                        button_song.play()
                        menu = True
                        choose_selection = False
                else:
                    choose_exit_button_time = 0
            # SELECTION END

            # GAME START
            if game_mode:

                # exit button
                image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/frame3_easy.png')
                exit_game = Button([10, 20], '', [111, 111], (50, 50, 100))
                if exit_game.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/frame3_easy_exit.png')
                    exit_game_button_time += 1
                    if exit_game_button_time == button_pause:
                        button_song.play()
                        menu = True
                        game_mode = False
                        round_index = 0
                        alarm_song.pause()
                else:
                    exit_game_button_time = 0

                # Conditions for hand correction
                if hand_correction:
                    cv2.rectangle(image, (0, 0), (170, 35), (245, 117, 16), -1)
                    l_well = pose_correct(image, 'Straighten your left arm!', angle_left_elbow, 160, 10)
                    r_well = pose_correct(image, 'Straighten your right arm!', angle_left_elbow, 160, 25)
                    hand_corr = l_well * r_well
                else:
                    hand_corr = 1

                # Conditions for body correction
                if posture_correction:
                    cv2.rectangle(image, (0, 0), (170, 35), (245, 117, 16), -1)
                    posture = pose_correct(image, 'Keep your head straight!', angle_median, 105, 10, 75)
                else:
                    posture = 1

                hour, minute = round_time[round_index] # time from generated array

                # Start game
                if round_index != round_len:
                    cv2.putText(image, str(round_done) + '/' + str(round_len), (500, 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

                    cv2.putText(image, str(hour) + ':' + str(minute * 5), (610, 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

                    ready_now = time.time()
                    delta = ready_now - ready_start

                    # 3...2...1..
                    if ready_flag:
                        ready, image = ready_strt(delta, image, d_time=d_time)
                        delta1 = delta
                        if ready:
                            ready_flag = 0
                    elif easy_game:
                        sec = delta - delta1
                        all_time = time.strftime("%M:%S", time.gmtime(sec))
                        cv2.rectangle(image, (200, 30), (350, 80), (89, 92, 87), -1)
                        cv2.putText(image, all_time, (210, 70),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        sec_hard_start = abs(delta - delta1)
                        time_left = hard_game_round_time - sec_hard_start
                        all_time = time.strftime("%M:%S", time.gmtime(time_left))
                        cv2.rectangle(image, (200, 30), (350, 80), (89, 92, 87), -1)
                        cv2.putText(image, all_time, (210, 70),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

                    # Main function
                    try:
                        if ready and detect(image, hour, minute, median[0], median[1], nose[0], nose[1], right_wrist[0],
                                  right_wrist[1], left_wrist[0], left_wrist[1], game_mode, draw_arrow) and \
                                hand_corr and posture:
                            cv2.rectangle(image, (480, 10), (800, 90), (175, 255, 46), -1)
                            cv2.putText(image, str(hour) + ':' + str(minute * 5), (610, 80),
                                         cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
                            time_index += 1
                        else:
                            time_index = 0


                        if time_index == time_step:
                            round_done += 1
                            sound_win.play()
                            round_index += 1
                            ready_start = time.time()
                            hard_game_round_time = const_hard_game_round_time
                            delta1 = 0
                        elif time_left < - 0.01:
                            round_index += 1
                            ready_start = time.time()
                            hard_game_round_time = const_hard_game_round_time
                            delta1 = 0
                    except:
                        if time_left < - 0.01:
                            round_index += 1
                            ready_start = time.time()
                            hard_game_round_time = const_hard_game_round_time
                            delta1 = 0

                else:
                    time_left = 1
                    ready_flag = 1
                    cv2.rectangle(image, (423, 216), (850, 450), (175, 255, 46), -1) 
                    cv2.putText(image, 'The End! ' + str(round_done) + '/' + str(round_len), (520, 300),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
                    if easy_game:
                        cv2.putText(image, 'Your Time: ' + all_time, (500, 350),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

            # GAME END

            # ALARM START
            if alarm_mode:
                # Exit button
                image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/frame3_easy.png')
                exit_game = Button([10, 20], '', [111, 111], (50, 50, 100))

                if exit_game.press(palm_r_x, palm_r_y, palm_l_x, palm_l_y):
                    image = segmentation(image, results.segmentation_mask, bg_img_path='imgs/frame3_easy_exit.png')
                    exit_game_button_time += 1
                    if exit_game_button_time == button_pause:
                        menu = True
                        alarm_mode = False
                        time_index_alarm = 0
                        alarm_song.pause()
                else:
                    exit_game_button_time = 0

                    # Conditins for hand correction
                if hand_correction:
                    cv2.rectangle(image, (0, 0), (170, 35), (245, 117, 16), -1)
                    l_well = pose_correct(image, 'Straighten your left arm!', angle_left_elbow, 160, 10)
                    r_well = pose_correct(image, 'Straighten your right arm!', angle_left_elbow, 160, 25)
                    hand_corr = l_well * r_well
                else:
                    hand_corr = 1

                # Conditins for body correction
                if posture_correction:
                    cv2.rectangle(image, (0, 0), (170, 35), (245, 117, 16), -1)
                    posture = pose_correct(image, 'Keep your head straight!', angle_median, 105, 10, 75)
                else:
                    posture = 1


                # Main function
                if time_index_alarm != time_step:
                    cv2.putText(image, str(hour_alarm) + ':' + str(minute_alarm * 5), (550, 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Please show this time!', (500, 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

                    if alarm_flag:
                        alarm_song.play()
                        alarm_flag = 0

                    if detect(image, hour_alarm, minute_alarm, median[0], median[1], nose[0], nose[1], right_wrist[0],
                                           right_wrist[1], left_wrist[0], left_wrist[1], game_mode, draw_arrow) and \
                            hand_corr and posture:
                        time_index_alarm += 1
                    else:
                        time_index_alarm = 0

                else:
                    cv2.rectangle(image, (500, 200), (900, 400), (0, 255, 0), -1)
                    cv2.putText(image, 'Good job! Good song', (520, 300),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
                    alarm_song.pause()



            # ALARM END
        except:
            pass

        # draw landmarks
        if landmarks_show:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow(window_name, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
