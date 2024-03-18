import os
import cv2
import time
import numpy as np
import pandas as pd
import seaborn as sns
from imutils import paths
from tqdm.notebook import tqdm

from scipy.signal import convolve2d
from skimage import measure

from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class Image_Classifier_Model:
    def __init__(self) -> None:
        self.input_size_convolve_img = (32, 32)
        
    def convolve_img(self, array, filters: int= 32, kernel_size: np.array=(5,5), padding: str= 'same', activation='relu', seend: int= 51972):
        np.random.seed(seend)
        convolved_array = []
    
        if isinstance(array, np.ndarray) and array.shape == (self.input_size_convolve_img[0], self.input_size_convolve_img[1], 3):
            array = cv2.split(array)
            
        for layer in array:
            for i in range(filters):
                filter = np.random.rand(kernel_size[0], kernel_size[1])
                convolved_layer = convolve2d(layer, filter, padding)
                
                if activation == 'relu':
                    activated_layer = np.where(convolved_layer < 0, 0, convolved_layer)
                convolved_array.append(activated_layer)
    
        return convolved_array

    def max_pooling_img(self, array, kernel_size: np.array=(2,2)):
        pooled_array = []
        for layer in array:
            pooled_layer = measure.block_reduce(layer, kernel_size, np.max)
            pooled_array.append(pooled_layer)
        return pooled_array

    def dropout(self, array, index: str= 'even'):
        even_array = []
        odd_array = []
        for i in range(len(array)):
            if (i+1) % 2 == 0:
                even_array.append(array[i])
            else:
                odd_array.append(array[i])
                
        if index == 'even': return even_array
        return odd_array
    
    def flatten(self, array):
        flattened_array = []
        for value in array:
            row, column = value.shape
            for i in range(row):
                for j in range(column):
                    flattened_array.append(value[i, j])

        return flattened_array
    
    def extract_feature(self, img):
        img = cv2.resize(img, self.input_size_convolve_img)
        img = img.astype('float32') / 255.0
        
        convolved_array_1 = self.convolve_img(img, filters= 16)
        pooled_array_1 = self.max_pooling_img(convolved_array_1)
        dropout_array_1 = self.dropout(pooled_array_1, index= 'even')
        
        convolved_array_2 = self.convolve_img(dropout_array_1, filters= 8, padding='valid')
        pooled_array_2 = self.max_pooling_img(convolved_array_2)
        dropout_array_2 = self.dropout(pooled_array_2, index= 'odd')
        
        convolved_array_3 = self.convolve_img(dropout_array_2, filters= 8, padding='valid')
        pooled_array_3 = self.max_pooling_img(convolved_array_3)
        dropout_array_3 = self.dropout(pooled_array_3, index= 'even')
        
        flattened_array = self.flatten(dropout_array_3)
        return flattened_array

    def img_augmentation(self, img, horizontal_flip=False, vertical_flip=False, brightness_range=[0.9, 1.1], zoom_range=0.1, shift_range=0.2, channel_shift_range=0.5):
        def _horizontal_flip(img):
            return cv2.flip(img, 1)
    
        def _vertical_flip(img):
            return cv2.flip(img, 0)
    
        def _change_brightness(img, brightness_range, num_of_results= 5):
            results = []
            for i in range(num_of_results):
                channel_shift_range = 0.5
                channel_b, channel_g, channel_r = cv2.split(img)
                channel_b = channel_b.astype(np.float32)
                channel_g = channel_g.astype(np.float32)
                channel_r = channel_r.astype(np.float32)
                channel_b += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                channel_g += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                channel_r += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                img = cv2.merge([channel_b, channel_g, channel_r])
                img = np.clip(img, 0, 255).astype(np.uint8)
                results.append(img)
            return results
            
        def _zoom(img, zoom_range=5, num_of_results= 5):
            results = []
            for i in range(num_of_results):
                zoom_factor = np.random.uniform(1-zoom_range, 1+zoom_range)
                height, width = img.shape[:2]
                resized_image = cv2.resize(img, (int(width * zoom_factor), int(height * zoom_factor)))
                results.append(resized_image)
            return results
            
        def _shift(img, shift_range= 0.2, num_of_results= 5):
            results = []
            height, width = img.shape[:2]
            for i in range(num_of_results):
                height_shift = np.random.uniform(-shift_range, shift_range) * height
                width_shift = np.random.uniform(-shift_range, shift_range) * width
                translation_matrix = np.float32([[1, 0, width_shift], [0, 1, height_shift]])
                img = cv2.warpAffine(img, translation_matrix, (width, height))
                results.append(img)
            return results
    
        def _channel_shift(img, channel_shift_range= 0.5, num_of_results= 5):
            results = []
            for i in range(num_of_results):
                channel_b, channel_g, channel_r = cv2.split(img)
                channel_b = channel_b.astype(np.float32)
                channel_g = channel_g.astype(np.float32)
                channel_r = channel_r.astype(np.float32)
                channel_b += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                channel_g += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                channel_r += np.random.uniform(-channel_shift_range, channel_shift_range) * 255
                img = cv2.merge([channel_b, channel_g, channel_r])
                img = np.clip(img, 0, 255).astype(np.uint8)
                results.append(img)
            return results
    
        results = [img]
        
        if horizontal_flip:
            results.append(_horizontal_flip(img))
        if vertical_flip:
            results.append(_vertical_flip(img))
        if brightness_range != None:
            results.extend(_change_brightness(img, brightness_range, num_of_results= 1))
        if zoom_range != None:
            results.extend(_zoom(img, zoom_range, num_of_results= 1))
        if shift_range != None:
            results.extend(_shift(img, shift_range, num_of_results= 1))
        if channel_shift_range != None:
            results.extend(_channel_shift(img, channel_shift_range, num_of_results= 1))
    
        return results

    def chunk_list(self, lst, n):
        size = len(lst) // n
        chunks = [lst[i * size:(i + 1) * size] for i in range(n - 1)]
        chunks.append(lst[(n - 1) * size:])
        return chunks
    
    def load_data(self, path):
        train_df, test_df, self.classes, self.class_name_to_int, self.int_to_class_name, self.input_size_convolve_img = load(path)
        return train_df, test_df
    
    def save_data(self,train_df, test_df, path):
        dump_file = [train_df, test_df, self.classes, self.class_name_to_int, self.int_to_class_name, self.input_size_convolve_img]
        dump(dump_file, path)
    
    def get_class(self, path):
        self.classes = os.listdir(path)
        self.class_name_to_int = dict(zip(self.classes, range(len(self.classes))))
        self.int_to_class_name = dict(zip(range(len(self.classes)), self.classes))

    def save_model(self, model_name: str = 'classifier', output_folder: str = './runs'):
        self.create_result_file(output_folder)

        dump_file = [self.model, self.classes, self.class_name_to_int, self.int_to_class_name, self.input_size_convolve_img]
        self.ax.figure.savefig(f'{self.folder_name}/Confusion matrix.png')
        print(f'Save in "{self.folder_name}/{model_name}.model"')
        dump(dump_file,f'{self.folder_name}/{model_name}.model')

    def load_model(self, path):
        self.model, self.classes, self.class_name_to_int, self.int_to_class_name, self.input_size_convolve_img = load(path)
        print('Done!')

    def create_result_file(self, output_folder):
        num_file = 0    
        folder_name = '{}/classifier_{}'
        try:
            os.makedirs(output_folder)
        except: pass
        while True:
            try:
                self.folder_name = folder_name.format(output_folder, num_file)
                os.makedirs(self.folder_name)
                break
            except: 
                num_file += 1 
    
    def data_generator(self, path, horizontal_flip=False, vertical_flip=False, brightness_range= None, zoom_range= None, shift_range= None, channel_shift_range= None):
        '''
        return: list([img_1, num_class_1], [img_2, num_class_2], ...)
        '''
        self.get_class(path)
        
        all_img_paths = list(paths.list_images("."))
        img_paths = []
        for img_path in all_img_paths:
            if img_path.startswith(path):
                img_paths.append(img_path)
    
        len_before = len(img_paths)
        results = []
        
        for i in tqdm(range(len(img_paths)), desc="Progress"):
            img_path = img_paths[i]
    
            _, _, _, class_name, _ = img_path.split('\\')
            
            img = cv2.imread(img_path)
            
            generated_img = self.img_augmentation(img, horizontal_flip, vertical_flip, brightness_range, zoom_range, shift_range, channel_shift_range)
    
            img_and_class = list( zip(generated_img, [self.class_name_to_int[class_name]]*(len(generated_img))))
            
            results.extend(img_and_class)
    
        len_after = len(results)
    
        print(f'{len_before} ===> {len_after}')
        return results
     
    def convert_imgData_to_featureData(self, data: list, size: tuple= (32, 32)) -> pd.DataFrame:
        '''
        data = list([img_1, num_class_1], [img_2, num_class_2], ...)
        return df
        '''
        self.input_size_convolve_img = size
        data_for_df = []
        for i in tqdm(range(len(data)), desc="Progress"):
            img, class_num = data[i]
            feature = self.extract_feature(img)
            
            row = feature.copy()
            row.append(class_num)
            data_for_df.append(row)
            
        df = pd.DataFrame(data_for_df)
        df = df.rename(columns={df.columns[-1]: 'class'})
        return df
     
    def train(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        X_train = df_train.drop('class', axis=1)
        y_train = df_train['class']
    
        X_test = df_test.drop('class', axis=1)
        y_test = df_test['class']
    
        ros = RandomOverSampler(random_state=42)
        rus = RandomUnderSampler(random_state=42)
        X_train_ros_resampled, y_train_ros_resampled = ros.fit_resample(X_train, y_train)
        X_test_rus_resampled, y_test_rus_resampled = rus.fit_resample(X_test, y_test)
        
        self.model = svm.SVC(C= 22.871)
        self.model.fit(X_train_ros_resampled, y_train_ros_resampled)
        
        y_pred = self.model.predict(X_test_rus_resampled)
        
        conf_matrix = confusion_matrix(y_test_rus_resampled, y_pred)
        normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(normalized_conf_matrix, index= self.classes, columns= self.classes)
        
        self.ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='hot', cbar=True)

    def predict(self, img):
        start_time = time.time()
        feature = []
        feature.append(self.extract_feature(img))

        result = self.model.predict(feature)[0]
        class_name = self.int_to_class_name[result]
        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        print(f'Time: {execution_time}s\nResult: {class_name}')
        return class_name