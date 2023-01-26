import os
import cv2
import dlib
import pandas as pd

from imutils.face_utils import rect_to_bb

from enum import Enum

from src.experiment.data_loading.local_data_loader import Dataset, LocalDataLoader, DatasetSplit

DATA_LOADER = LocalDataLoader()
FACE_DETECTOR = dlib.get_frontal_face_detector()


def prep_df_windows_paths(df):
    df['origin'] = df['Nome'].apply(lambda x : x.split('\\')[-4])
    df['split'] = df['Nome'].apply(lambda x : x.split('\\')[-3])
    df['label'] = df['Nome'].apply(lambda x : x.split('\\')[-2])
    df['img_name'] = df['Nome'].apply(lambda x: x.split('\\')[-1])
    return df

def prep_df_ubuntu_paths(df):
    df['origin'] = df['Nome'].apply(lambda x : x.split('/')[-4])
    df['split'] = df['Nome'].apply(lambda x : x.split('/')[-3])
    df['label'] = df['Nome'].apply(lambda x : x.split('/')[-2])
    df['img_name'] = df['Nome'].apply(lambda x: x.split('/')[-1])
    return df

class DetectionState(Enum):
    NO_FACES = 'No faces found'
    FACE_FOUND = 'Face found'
    MULTI_FACES_FOUND = 'Multi faces'

    
def fixed_resize_img(img):
    h,w,_ = img.shape
    proportion = -1
    if h > w:
        proportion = 640./h
    else:
        proportion = 640./w
    new_h, new_w = int(h*proportion),int(w*proportion)
    img = cv2.resize(img, (new_w,new_h))
    return img, proportion
    
    
def conditional_resize_img(img):
    h,w,_ = img.shape
    proportion = -1
    if(h>640 or w>640):
        proportion = None
        if h > w:
            proportion = 640./h
        else:
            proportion = 640./w
        new_h, new_w = int(h*proportion),int(w*proportion)
        img = cv2.resize(img, (new_w,new_h))
    return img, proportion

    
def run_conditional_face_detector(img_path, upsample):
    img = cv2.imread(img_path)
    img, proportion = conditional_resize_img(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR(img_gray, upsample)
    n_faces = len(faces)
    if n_faces == 1:
        bbox = rect_to_bb(faces[0])
        return (DetectionState.FACE_FOUND, [bbox], proportion)
    elif n_faces > 1:
        bboxes = []
        for i in range(len(faces)):
            bboxes.append(rect_to_bb(faces[i]))
        return (DetectionState.MULTI_FACES_FOUND, bboxes, proportion)
    elif n_faces == 0:
        return (DetectionState.NO_FACES, [], proportion)
    else:
        raise Exception("Sorry, no numbers below zero")

def run_fixed_face_detector(img_path, upsample):
    img = cv2.imread(img_path)
    img, proportion = fixed_resize_img(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR(img_gray, upsample)
    n_faces = len(faces)
    if n_faces == 1:
        bbox = rect_to_bb(faces[0])
        return (DetectionState.FACE_FOUND, [bbox], proportion)
    elif n_faces > 1:
        bboxes = []
        for i in range(len(faces)):
            bboxes.append(rect_to_bb(faces[i]))
        return (DetectionState.MULTI_FACES_FOUND, bboxes, proportion)
    else:
        return (DetectionState.NO_FACES, [], proportion)




def get_dataset_by_origin(input_df, origin_dataset):
    df = input_df[input_df.origin == origin_dataset.value]
    print(f'{origin_dataset.value.upper()} shape: {df.shape}')
    print(f'{origin_dataset.value.upper()} #img_paths: {len(df.Nome.unique())}')
    print(f'{origin_dataset.value.upper()} splits: {df.split.unique()}')
    print('----------------------------------------')
    return df




def load_dataset(dataset):
    final_df = pd.DataFrame()
    
    DATA_LOADER.load_data([dataset], DatasetSplit.TRAIN)
    DATA_LOADER.load_data([dataset], DatasetSplit.DEVEL)
    DATA_LOADER.load_data([dataset], DatasetSplit.TEST)
    
    final_df = pd.concat([DATA_LOADER.train_data_df, DATA_LOADER.devel_data_df, DATA_LOADER.test_data_df], ignore_index=True)
    print(f'{dataset.value.upper()} shape: {final_df.shape}')
    return final_df


def prep_df(df):
    df['origin'] = df['Nome'].apply(lambda x : x.split('\\')[-4])
    df['split'] = df['Nome'].apply(lambda x : x.split('\\')[-3])
    df['label'] = df['Nome'].apply(lambda x : x.split('\\')[-2])
    df['img_name'] = df['Nome'].apply(lambda x: x.split('\\')[-1])
    return df

def prep_df_2(df):
    df['origin'] = df['img_path'].apply(lambda x : x.split(os.path.sep)[-4])
    df['split'] = df['img_path'].apply(lambda x : x.split(os.path.sep)[-3])
    df['label'] = df['img_path'].apply(lambda x : x.split(os.path.sep)[-2])
    df['img_name'] = df['img_path'].apply(lambda x: x.split(os.path.sep)[-1])
    return df


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou