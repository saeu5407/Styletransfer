import os
import glob
import cv2 as cv
import pandas as pd
import random

def mitty_divide_image_labels(data_path, name, video_idx, label_csv):

    # Make Dir
    make_dir(path=os.path.join(data_path, name))
    for dir in label_csv['label'].unique():
        make_dir(os.path.join(data_path, name, dir))

    # Make Labeled Image Data
    cap = cv.VideoCapture(video_idx)

    index = 0
    token = True
    while True:

        ret, frame = cap.read()
        index += 1

        if index % 10000 == 0:
            print('>>> Index {} Save'.format(index))

        if not ret or not token:
            print('>>> Save Image | {}'.format(name))
            print('>>> Final Index is | {}'.format(index))
            cap.release()
            break

        # Labeling & Save Images
        labels = label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'label']
        idx_len = label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'end_index'] - \
                  label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'start_index']

        if len(labels) > 0:
            if idx_len.values[0] >= 1000:
                if index % 50 == 0:
                    label_path = os.path.join(data_path, name, labels.values[0], str(index) + '.png')
                    cv.imwrite(label_path, frame)
            else:
                label_path = os.path.join(data_path, name, labels.values[0], str(index) + '.png')
                cv.imwrite(label_path, frame)

        # Quit if not labeled on idx_data
        if label_csv.end_index.max() <= index:
            token = False


def mitty_image_labels(data_path, name, video_idx, label_csv):

    # Make Dir
    make_dir(os.path.join(data_path, 'image'))
    for dir in label_csv['label'].unique():
        make_dir(os.path.join(data_path, 'image', dir))

    # Make Labeled Image Data
    cap = cv.VideoCapture(video_idx)

    index = 0
    token = True
    while True:

        ret, frame = cap.read()
        index += 1

        if index % 10000 == 0:
            print('>>> Index {} Save'.format(index))

        if not ret or not token:
            print('>>> Save Image | {}'.format(name))
            print('>>> Final Index is | {}'.format(index))
            cap.release()
            break

        # Labeling & Save Images
        labels = label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'label']
        idx_len = label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'end_index'] - \
                  label_csv.loc[(label_csv.start_index <= index) & (label_csv.end_index >= index), 'start_index']

        if len(labels) > 0:
            if idx_len.values[0] >= 1000:
                if index % 10 == 0:
                    label_path = os.path.join(data_path, 'image', labels.values[0], name + '__' + str(index) + '.png')
                    cv.imwrite(label_path, frame)
            elif idx_len.values[0] >= 500:
                if index % 5 == 0:
                    label_path = os.path.join(data_path, 'image', labels.values[0], name + '__' + str(index) + '.png')
                    cv.imwrite(label_path, frame)
            else:
                label_path = os.path.join(data_path, 'image', labels.values[0], name + '__' + str(index) + '.png')
                cv.imwrite(label_path, frame)

        # Quit if not labeled on idx_data
        if label_csv.end_index.max() <= index:
            token = False


def make_dir(path):

    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':

    # Data Path
    data_path = os.path.join(os.getcwd().split('/src')[0], 'datasets', 'mitty')
    print(data_path)

    # Find Video, Label Data
    label_list = glob.glob(os.path.join(data_path, 'labels', '*.csv'))
    video_list = glob.glob(os.path.join(data_path, 'video', '*.mp4'))

    # Loop
    for idx in range(0, len(label_list)):

        # Load Labels
        print(label_list[idx])
        name = os.path.basename(label_list[idx]).split('.csv')[0]
        try:
            label_csv = pd.read_csv(label_list[idx])
        except:
            label_csv = pd.read_csv(label_list[idx], encoding='cp949')

        # Load Video
        video_idx = 0
        for videoidx in video_list:
            if videoidx.find(str(name)) != -1:
                video_idx = videoidx

        if video_idx != 0:
            mitty_image_labels(data_path, name, video_idx, label_csv)