import os
import pathlib
import pandas as pd
import numpy as np

from enum import Enum

from src.configs.config import BASE_PATH

class Dataset(Enum):
    NUAA = 'NUAA'
    MSU = 'MSU'
    REPLAY_ATTACK = 'ReplayAttack'
    OULU = 'OULU'


class DatasetSplit(Enum):
    TRAIN = 'train'
    DEVEL = 'devel'
    TEST = 'test'



class LocalDataLoader:
    def __init__(self, dataset_version):
        self.DATASETS_ROOT_DIR = os.path.join(BASE_PATH, dataset_version.name)

        print(f'----')
        print(f'DATASETS_ROOT_DIR: {self.DATASETS_ROOT_DIR}')
        print(f'----')

        self.train_output_path = os.path.join(f'{self.DATASETS_ROOT_DIR}', 'train_data.csv')
        self.devel_output_path = os.path.join(f'{self.DATASETS_ROOT_DIR}', 'devel_data.csv')
        self.test_output_path = os.path.join(f'{self.DATASETS_ROOT_DIR}', 'test_data.csv')

        self.train_data_df = None
        self.devel_data_df = None
        self.test_data_df = None
        
        self.train_dirs_paths = [os.path.join(self.DATASETS_ROOT_DIR, x.value, DatasetSplit.TRAIN.value) for x in Dataset]
        self.devel_dirs_paths = [os.path.join(self.DATASETS_ROOT_DIR, x.value, DatasetSplit.DEVEL.value) for x in Dataset]
        self.test_dirs_paths = [os.path.join(self.DATASETS_ROOT_DIR, x.value, DatasetSplit.TEST.value) for x in Dataset]

    
    def __is_valid_imgpath(self, imgpath):
        return '.JPG' in imgpath or '.PNG' in imgpath \
                or '.jpg' in imgpath or '.png' in imgpath \
                or '.ppm' in imgpath
    
    def getLabel(self, row):
        img_name = row[0]
        return row['img_path'].split(os.path.sep)[-2]
            
    def getOrigin(self, row):
        img_name = row[0]
        return row['img_path'].split(os.path.sep)[-4]
    
    def getSplit(self, row):
        img_name = row[0]
        return row['img_path'].split(os.path.sep)[-3]
    
    def getImgName(self, row):
        img_name = row[0]
        return row['img_path'].split(os.path.sep)[-1].split('##')[1]
    
    def getUID(self, row):
        return row['img_path'].split(os.path.sep)[-1].split('##')[0]

            
    def __create_data_df_subsampled(self, input_dir, total_subsampled, input_file, output_file_path):
        origins = list(set([origin.split("\\")[4] for origin in input_dir]))
        splits = list(set([s.split("\\")[5] for s in input_dir]))
        labels = ['attack', 'real']
        df = pd.read_csv(input_file)
        for origin in origins:
            for split in splits:
                for label in labels:
                    df_aux =  df[(df['split']==split) & (df['label']==label) & (df['origin']==origin)]
                    total_clients = np.unique(df_aux['image_name_1'].str.split('_').str[1])
                    images_per_client = total_subsampled/len(total_clients)
                    for client in total_clients:
                        df_ = df_aux[df_aux['image_name_1'].str.contains(client)].sort_values(by=['ssim_score'])[:int(images_per_client)]
                        if not os.path.isfile(output_file_path):
                            df_[['image_name_2', 'label','origin']].to_csv(output_file_path, mode='a', header=["img_path","label","origin"], index=False)
                        else:
                            df_[['image_name_2', 'label','origin']].to_csv(output_file_path, mode='a', header=False, index=False)
       
                            
    def __create_data_df(self, split, input_dir, output_path):
        files_list = []
        
        for d in input_dir:
            for path, _, files in os.walk(d):
                for name in files:
                    p = str(pathlib.PurePath(path, name))
                    if self.__is_valid_imgpath(p):
                        files_list.append(p)
        
        files_list = sorted(files_list)
        
        print(f'..total number of {split.value.upper()} images: {len(files_list)}')
        
        data_df = pd.DataFrame()
        data_df['img_path'] = files_list
        data_df['label'] = data_df.apply(lambda row : self.getLabel(row), axis=1)
        data_df['origin'] = data_df.apply(lambda row : self.getOrigin(row), axis=1)
        data_df['split'] = data_df.apply(lambda row : self.getSplit(row), axis=1)
        data_df['img_name'] = data_df.apply(lambda row : self.getImgName(row), axis=1)
        data_df['uid'] = data_df.apply(lambda row: self.getUID(row), axis=1)
        
        data_df.to_csv(output_path, index=False)
    
    
    def setup_data_df(self):
        print('\nSetup dataset...')

        self.__create_data_df(split=DatasetSplit.TRAIN, input_dir=self.train_dirs_paths, output_path=self.train_output_path)
        self.__create_data_df(split=DatasetSplit.DEVEL, input_dir=self.devel_dirs_paths, output_path=self.devel_output_path)
        self.__create_data_df(split=DatasetSplit.TEST, input_dir=self.test_dirs_paths, output_path=self.test_output_path)
        
        #self.__create_data_df_subsampled(self.test_dirs_paths, 6000, 'preprocessing/subsampled_ssim_file.csv', self.test_output_path)
        #print('Images names dataframe saved')

        print('Dataset setup done!\n')
    
    
    def load_data(self, datasets_list, split):
        if split.value not in [DatasetSplit.TRAIN.value, DatasetSplit.DEVEL.value, DatasetSplit.TEST.value]:
            print('Invalid dataset split!')
            return
        
        if split.value == DatasetSplit.TRAIN.value:
            print(f' ..train_output_path: {self.train_output_path}')
            if os.path.exists(self.train_output_path):
                self.train_data_df = pd.read_csv(self.train_output_path)
                self.train_data_df = self.train_data_df[self.train_data_df.origin.isin([x.value for x in datasets_list])]
            else:
                print('Train data csv does not exist!')
        
        if split.value == DatasetSplit.DEVEL.value:
            print(f' ..devel_output_path: {self.devel_output_path}')
            if os.path.exists(self.devel_output_path):
                self.devel_data_df = pd.read_csv(self.devel_output_path)
                self.devel_data_df = self.devel_data_df[self.devel_data_df.origin.isin([x.value for x in datasets_list])]
            else:
                print('Devel data csv does not exist!')
        
        if split.value == DatasetSplit.TEST.value:
            print(f' ..test_output_path: {self.test_output_path}')
            if os.path.exists(self.test_output_path):
                self.test_data_df = pd.read_csv(self.test_output_path)
                self.test_data_df = self.test_data_df[self.test_data_df.origin.isin([x.value for x in datasets_list])]
            else:
                print('Test data csv does not exist!')
            
    
    def __count_image_per_class(self, origin, split):
        data_df = None
        if split == DatasetSplit.TRAIN:
            data_df = self.train_data_df
        elif split == DatasetSplit.DEVEL:
            data_df = self.devel_data_df
        elif split == DatasetSplit.TEST:
            data_df = self.test_data_df
        
        n_attack = data_df[(data_df.label == 'attack') & (data_df.origin == origin)].shape[0]
        n_real = data_df[(data_df.label == 'real') & (data_df.origin == origin)].shape[0]
        total = n_attack + n_real
        alldata_total = data_df.shape[0]
        
        print(f'N_ATTACK: {n_attack} ({round(n_attack/total * 100, 2)}%)')
        print(f'N_REAL: {n_real} ({round(n_real/total * 100, 2)}%)')
        print(f'TOTAL: {total}/{alldata_total} ({round(total/alldata_total* 100, 2)}%)')
    
    
    def summary_data(self, datasets_list):
        splits = [DatasetSplit.TRAIN, DatasetSplit.DEVEL, DatasetSplit.TEST]
        
        for spt in splits:
            print('=======================')
            for dataset in datasets_list:
                dataset = dataset.value
                if not os.path.exists(os.path.join(self.DATASETS_ROOT_DIR, 'imgs', dataset, spt.value)):
                    print('--------------')
                    print(f"No {dataset} {spt.value.upper()} split")
                    continue
                
                print('--------------')
                print(f'Dataset: {dataset} - {spt.value.upper()}')
                self.__count_image_per_class(dataset, spt)
            