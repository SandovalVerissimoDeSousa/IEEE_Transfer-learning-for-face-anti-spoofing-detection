import os
import cv2
import ffmpeg

from glob import glob
from enum import Enum
from operator import add
from pathlib import Path
from shutil import copyfile

from config import DATASETS_ROOT_DIR
from config import ORIGINAL_DATASETS_DIR



def CountFrames(DatasetName, total_real_frames, total_attack_frames):  
    
    if DatasetName == Dataset.MSU.value:
        true_frames = [0 , 0, 0]
        attack_frames = [0, 0, 0]
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\train\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\train\\*.mp4'))
        for file in filenames:
            #print(file)
            video = cv2.VideoCapture(file)
            #print(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
            true_frames[0] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 1/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\train\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\train\\*.mp4'))
        for file in filenames:
            video = cv2.VideoCapture(file)
            attack_frames[0] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 2/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\test\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\test\\*.mp4'))
        for file in filenames:
            video = cv2.VideoCapture(file)
            true_frames[2] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 3/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\test\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\test\\*.mp4'))
        for file in filenames:
            video = cv2.VideoCapture(file)
            attack_frames[2] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 4/4")
        print("MSU True Frames: ",true_frames)
        print("MSU Attack Frames: ",attack_frames)
        for i in range(3):
            total_real_frames[i] += true_frames[i]
            total_attack_frames[i] += attack_frames[i]
        print("Total True Frames: ",total_real_frames)
        print("Total Attack Frames: ",total_attack_frames)
              
    if DatasetName == Dataset.REPLAY_ATTACK.value:
        true_frames = [0 , 0, 0]
        attack_frames = [0, 0, 0]
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\train\\real\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            true_frames[0] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 1/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\train\\attack\\*\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            attack_frames[0] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 2/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\devel\\real\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            true_frames[1] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 3/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\devel\\attack\\*\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            attack_frames[1] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 4/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\test\\real\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            true_frames[2] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 5/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\test\\attack\\*\\*.mov')
        for file in filenames:
            video = cv2.VideoCapture(file)
            attack_frames[2] += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        print("Done! 6/6")
        print("Repaly Attack True Frames: ",true_frames)
        print("Repaly Attack Attack Frames: ",attack_frames)
        for i in range(3):
            total_real_frames[i] += true_frames[i]
            total_attack_frames[i] += attack_frames[i]
        print("Total True Frames: ",total_real_frames)
        print("Total Attack Frames: ",total_attack_frames)

    if DatasetName == Dataset.NUAA.value:
        true_frames = [0 , 0, 0]
        attack_frames = [0, 0, 0]
        filenames = []
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\client_train_raw.txt", "r")
        for filepath in index_file:
            filenames.append(DATASETS_ROOT_DIR + "NUAA\\raw\\" + filepath)
        true_frames[0] = len(filenames)
        print("Done! 1/4")
        filenames = []
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\imposter_train_raw.txt", "r")
        for filepath in index_file:
            filenames.append(DATASETS_ROOT_DIR + "NUAA\\raw\\" + filepath)
        attack_frames[0] = len(filenames)
        print("Done! 2/4")
        filenames = []
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\client_test_raw.txt", "r")
        for filepath in index_file:
            filenames.append(DATASETS_ROOT_DIR + "NUAA\\raw\\" + filepath)
        true_frames[2] = len(filenames)
        print("Done! 3/4")
        filenames = []
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\imposter_test_raw.txt", "r")
        for filepath in index_file:
            filenames.append(DATASETS_ROOT_DIR + "NUAA\\raw\\" + filepath)
        attack_frames[2] = len(filenames)
        print("Done! 4/4")
        
        print("NUAA True Frames: ",true_frames)
        print("NUAA Attack Frames: ",attack_frames)
        
        for i in range(3):
            total_real_frames[i] += true_frames[i]
            total_attack_frames[i] += attack_frames[i]
            
        print("Total True Frames: ",total_real_frames)
        print("Total Attack Frames: ",total_attack_frames)

        
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_code = None
    rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
    return round(int(rotate) / 90.0) * 90


def ExtractFrames(InputFile, OutputDir, Factor = 1, Prefix = ''):
    base = os.path.basename(InputFile)
    InputFile_name = os.path.splitext(base)[0]
    #Rotation = check_rotation(InputFile)
    #print(InputFile, " - ", Rotation)
    vidcap = cv2.VideoCapture(InputFile)
    success,image = vidcap.read()
    count = 0
    
    while success:
        if (count % int(Factor)) == 0:
        #     #if Rotation != 0:
        #         #image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(OutputDir + os.sep + InputFile_name + "_"+ Prefix + "-%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

        
def ProcessDataset (DatasetName, Factor = 1, Clear = False):
    if DatasetName == Dataset.MSU.value:
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\train\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\train\\*.mp4'))
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\train\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\train\\real"
        for file in filenames:
            #print("File: ", file)
            #print("OutputDir: ", OutputDir)
            ExtractFrames(file, OutputDir, Factor)      
        print("Done! 1/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\train\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\train\\*.mp4'))
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\train\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\train\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 2/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\test\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\real\\test\\*.mp4'))
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\test\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\test\\real"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 3/4")
        filenames = glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\test\\*.mov')
        filenames.extend(glob(DATASETS_ROOT_DIR + 'MSU-MFSD-Publish\\scene01\\attack\\test\\*.mp4'))
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\test\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\MSU\\test\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 4/4")
        
    if DatasetName == Dataset.REPLAY_ATTACK.value:
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\train\\real\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\real"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 1/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\train\\attack\\fixed\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Fixed')
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\train\\attack\\hand\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\train\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Hand')
        print("Done! 2/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\devel\\real\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\real"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 3/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\devel\\attack\\fixed\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Fixed')
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\devel\\attack\\hand\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\devel\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Hand')
        print("Done! 4/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\test\\real\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\real"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor)
        print("Done! 5/6")
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\test\\attack\\fixed\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Fixed')
        filenames = glob(DATASETS_ROOT_DIR + 'ReplayAttack\\test\\attack\\hand\\*.mov')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\ReplayAttack\\test\\attack"
        for file in filenames:
            ExtractFrames(file, OutputDir, Factor, 'Hand')
        print("Done! 6/6")
        
         
    if DatasetName == Dataset.ROSEYOUTU.value:

        dataset_path = os.path.join(ORIGINAL_DATASETS_DIR,DatasetName)
        filenames = [] # all mp4 files
        
        for folder in os.listdir(dataset_path): ## identities folders: 2,3,4,... etc
            for item in os.listdir(os.path.join(dataset_path,folder)):
                
                if item.endswith(".mp4"):
                    # print(item)
                    filenames.append(os.path.join(ORIGINAL_DATASETS_DIR,DatasetName,folder,item))
        lenght = len(filenames)
        print(str(lenght)+ ' \".mp4\" files found in all folders!')


        train_attack_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'train', 'attack')
        train_real_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'train', 'real')
        devel_attack_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'devel','attack')
        devel_real_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'devel', 'real')
        test_attack_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'test', 'attack')
        test_real_path = os.path.join(DATASETS_ROOT_DIR, DatasetName, 'test', 'real')
        Path(train_attack_path).mkdir(parents=True, exist_ok=True)
        Path(train_real_path).mkdir(parents=True, exist_ok=True)
        Path(devel_attack_path).mkdir(parents=True, exist_ok=True)
        Path(devel_real_path).mkdir(parents=True, exist_ok=True)
        Path(test_attack_path).mkdir(parents=True, exist_ok=True)
        Path(test_real_path).mkdir(parents=True, exist_ok=True)

        count = 0
        for file in filenames:
            id = int(file.split(os.sep)[-1].split('.')[0].split('_')[-2]) ## getting the ID (from 2 to 23)
            ground_truth = file.split(os.sep)[-1][0] ## 'G' means Genuine person, therefore its REAL and not ATTACK

            if int(id) in [2,3,4,5,6,7,9]: ############# train minus devel
                if ground_truth == "G": # real
                    ExtractFrames(file, train_real_path, Factor)
                else: # attack
                    ExtractFrames(file, train_attack_path, Factor)

            elif int(id) in [10,11,12]: ################ devel
                if ground_truth == "G": # real
                    ExtractFrames(file, devel_real_path, Factor)
                else: # attack
                    ExtractFrames(file, devel_attack_path, Factor)

            else: ################################ test
                if ground_truth == "G": # real
                    ExtractFrames(file, test_real_path, Factor)
                else: # attack
                    ExtractFrames(file, test_attack_path, Factor)
            
            print ("{:.2f}% Complete".format(count/lenght*100))
            count = count +1

        print("Done! 1/1")
        
        
    if DatasetName == Dataset.NUAA.value:
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\client_train_raw.txt", "r")
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\train\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\train\\real"
        for filepath in index_file:
            filepath = filepath.rstrip("\n")
            copyfile(DATASETS_ROOT_DIR + "NUAA\\raw\\real\\" + filepath, OutputDir + "\\" +filepath.split("\\")[1])
        print("Done! 1/4")
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\imposter_train_raw.txt", "r")
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\train\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\train\\attack"
        for filepath in index_file:
            filepath = filepath.rstrip("\n")
            copyfile(DATASETS_ROOT_DIR + "NUAA\\raw\\attack\\" + filepath, OutputDir + "\\" +filepath.split("\\")[1])
        print("Done! 2/4")
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\client_test_raw.txt", "r")
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\test\\real").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\test\\real"
        for filepath in index_file:
            filepath = filepath.rstrip("\n")
            copyfile(DATASETS_ROOT_DIR + "NUAA\\raw\\real\\" + filepath, OutputDir + "\\" +filepath.split("\\")[1])
        print("Done! 3/4")
        index_file = open(DATASETS_ROOT_DIR + "NUAA\\raw\\imposter_test_raw.txt", "r")
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\test\\attack").mkdir(parents=True, exist_ok=True)
        OutputDir = DATASETS_ROOT_DIR + "ProcessedFrames\\NUAA\\test\\attack"
        for filepath in index_file:
            filepath = filepath.rstrip("\n")
            copyfile(DATASETS_ROOT_DIR + "NUAA\\raw\\attack\\" + filepath, OutputDir + "\\" +filepath.split("\\")[1])
        print("Done! 4/4")
    
    if DatasetName == Dataset.OULU.value:
        filenames = glob(DATASETS_ROOT_DIR + 'RawOULU\\Test_files\\*.avi')
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\OULU\\test\\real").mkdir(parents=True, exist_ok=True)
        Path(DATASETS_ROOT_DIR + "ProcessedFrames\\OULU\\test\\attack").mkdir(parents=True, exist_ok=True)
        OutputDirTrue = DATASETS_ROOT_DIR + "ProcessedFrames\\OULU\\test\\real"
        OutputDirAttack = DATASETS_ROOT_DIR + "ProcessedFrames\\OULU\\test\\attack"
        count = 0
        print(len(filenames))
        for file in filenames:
            endfile = file.split('_')[-1]
            label = int(endfile.split('.')[0])
            #print(file)
            #print(os.path.isfile(file))
            if (label == 1):
                count += 1
                print(count)
                ExtractFrames(file, OutputDirTrue, Factor)
            else:
                ExtractFrames(file, OutputDirAttack, Factor)
                
        print("Done! 1/3")
        
              
def LoadDatabases(Databases, Factor = 1):
    train_true = []
    train_attack = []
    devel_true = []
    devel_attack = []
    test_true = []
    test_attack = []
    for DatasetName in Databases:
        if DatasetName == Dataset.MSU.value:
            print('MSU')
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\MSU\\train\\real\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    train_true.append(file)
            print("train_true: ", len(train_true))
            print("Done! 1/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\MSU\\train\\attack\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    train_attack.append(file)
            print("train_attack: ", len(train_attack))
            print("Done! 2/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\MSU\\test\\real\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    test_true.append(file)
            print("test_true: ", len(test_true))
            print("Done! 3/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\MSU\\test\\attack\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    test_attack.append(file)
            print("test_attack: ", len(test_attack))
            print("Done! 4/4")
            
        if DatasetName == Dataset.REPLAY_ATTACK.value:
            print('Replay')
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\train\\real\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    train_true.append(file)
            print("train_true: ", len(train_true))
            print("Done! 1/6")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\train\\attack\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    train_attack.append(file)
            print("train_attack: ", len(train_attack))
            print("Done! 2/6")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\devel\\real\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    devel_true.append(file)
            print("devel_true: ", len(devel_true))
            print("Done! 3/6")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\devel\\attack\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    devel_attack.append(file)
            print("devel_attack: ", len(devel_attack))
            print("Done! 4/6")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\test\\real\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    test_true.append(file)
            print("test_true: ", len(test_true))
            print("Done! 5/6")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\ReplayAttack\\test\\attack\\*')
            for file in filenames:
                index = int(file.split("-")[1].split(".")[0])
                if (index % Factor) == 0:
                    test_attack.append(file)
            print("test_attack: ", len(test_attack))
            print("Done! 6/6")
            
        if DatasetName == Dataset.NUAA.value:
            print('NUAA')
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\NUAA\\train\\real\\*')
            for file in filenames:
                train_true.append(file)
            print(len(train_true))
            print("Done! 1/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\NUAA\\train\\attack\\*')
            for file in filenames:
                train_attack.append(file)
            print(len(train_attack))
            print("Done! 2/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\NUAA\\test\\real\\*')
            for file in filenames:
                test_true.append(file)
            print(len(test_true))
            print("Done! 3/4")
            filenames = glob(DATASETS_ROOT_DIR + 'ProcessedFrames\\NUAA\\test\\attack\\*')
            for file in filenames:
                test_attack.append(file)
            print(len(test_attack))
            print("Done! 4/4")
            
    return train_true, train_attack, devel_true, devel_attack, test_true, test_attack