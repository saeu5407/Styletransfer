import os
import glob
import cv2
import pandas as pd
import random

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def make_image(data_path, video_path, label_name, index=0, save_seq=50):
    # Make Dir
    make_dir(path=os.path.join(data_path, label_name))

    # Make Image Data
    cap = cv2.VideoCapture(video_path)

    while True:

        ret, frame = cap.read()
        index += 1

        if index % 10000 == 0:
            print('>>> Index {} Save'.format(index))

        if not ret:
            print('>>> Save Image | {}'.format(label_name))
            print('>>> Final Index is | {}'.format(index))
            cap.release()
            break

        if index % save_seq == 0:
            label_path = os.path.join(data_path, label_name, str(label_name) + '_' + str(index) + '.png')
            cv2.imwrite(label_path, frame)

    return index

if __name__ == '__main__':

    # Data Path
    data_path = os.path.join(os.getcwd().split('/src')[0], 'datasets')

    # Find Video Data
    video_list = glob.glob(os.path.join(data_path, '*.mp4'))
    video_path = video_list[0]

    # Main
    idx = make_image(data_path=data_path, video_path=video_list[0], label_name='inf')
    make_image(data_path=data_path, video_path=video_list[1], label_name='inf', index=idx+1)
    make_image(data_path=data_path, video_path=video_list[2], label_name='ani', save_seq=20)
