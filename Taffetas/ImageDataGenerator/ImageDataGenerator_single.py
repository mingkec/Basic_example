from matplotlib import pyplot as plt
import numpy as np 
from os import listdir
import os
import tensorflow as tf
from tensorflow.contrib.keras import preprocessing
import cv2
class batch_data_generator:
    def __init__(self,image_size=(299,299),resize=False,num_threads=4,resize_method=1,
                 extension=False,cpu=False):
        self._test_file=[]
        self._resize_method=resize_method
        #resize_method= int
        '''
        BILINEAR = 0
        NEAREST_NEIGHBOR = 1
        BICUBIC = 2
        AREA = 3
        '''
        
        
        self.custom_function=False
        
        self.image_size=image_size
        self._resize=resize
        
        self.num_threads=num_threads
        
        self.extension=extension
        if extension is not False:
            if  type(extension) is not list:
                raise NameError('type of extension must be list')
         
        self._cpu=False
        
        self._init_tf_Session=False
        
        
    def _get_file_address(self,class_path,list_names):
        image_path=[]
        if self.extension == False:
            for names in list_names:            
                image_path.append(os.path.join(class_path,names))
        else:
            for names in list_names:
                for extension in self.extension:
                    #['.txt','.img','.jpeg']
                    if file.endswith(extension):
                        image_path.append(os.path.join(class_path,names))
          
        return image_path
   
    def _address_collection(self,filepath,merge=True,type=False):
        if type=='train':
            self.classes=sorted(listdir(filepath[0]))
            classes_dict={}
            for class_index in range(len(self.classes)):
                classes_dict.update({self.classes[class_index]:class_index})
            self.classes_dict=classes_dict
        
        number_of_classes=len(self.classes)
        number_of_locations=len(filepath)
        storage_file=[]
  
        for storage_file_index in range(number_of_locations):
            location_storage_file=[]
            for classes_index in range(number_of_classes):
                class_path=os.path.join(filepath[storage_file_index],self.classes[classes_index])
                #
                image_path=self._get_file_address(class_path,listdir(class_path))
                location_storage_file.append(image_path)
            storage_file.append(location_storage_file)      
        if merge == True:
            storage_file_reduce=[]
            for classes_index in range(number_of_classes):
                images_address=[]
                for storage_file_index in range(len(storage_file)):
                    images_address=images_address+storage_file[storage_file_index][classes_index]
                storage_file_reduce.append(images_address)
            storage_file=np.asarray(storage_file_reduce)
        else:
            storage_file=np.asarray(storage_file)
      
        return storage_file
    
    def _label_Initialization(self,storage_file,framework_format='one_hot',merge=True):
        if framework_format=='one_hot':
            if merge == True:
                storage_label=[]
                for class_ in range(len(storage_file)):
                    label=np.zeros([len(storage_file[class_]),len(storage_file)])
                    for index in range(len(label)):
                        label[index][class_]=1
                    storage_label.append(label)
            
            storage_label=np.asarray(storage_label)    
        else:
            raise NameError('framework_format must be one_hot')
        return storage_label
    
    def _batch_table_Initialization(self):
        #get class number
        batch_table=np.zeros(len(self.storage_file))
        
        for class_ in range(len(self.storage_file)):
            batch_table[class_]=len(self.storage_file[class_])
            
        return batch_table
    
    def _filepath_Initialization(self,filepath):
        if type(filepath) is str:
            self.filepath=[filepath]
        elif type(filepath) is list:
            self.filepath=filepath
        else:
            raise NameError('type of filepath must be str or list')
        
        
        self._train_path=self.filepath  
    
    def _tf_train_image_reader(self,image,label,image_size=(299,299),num_threads=4):

        filelength=len(image)
        
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.float32)
        input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
        label = input_queue[1]
        image_path=input_queue[0]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)
      

        image=tf.image.resize_images(image,image_size,method=self._resize_method)

        if self.custom_function!=False:
            
            image=tf.py_func(self.custom_function,[image],tf.float32)
            image.set_shape([image_size[0],image_size[1],3])


        capacity = filelength

        image_batch, label_batch, path_batch = tf.train.batch([image, label,image_path],
                                    batch_size=filelength,
                                    num_threads=1,
                                    capacity=capacity,
                                    enqueue_many=False,
                                    shapes=None,
                                    dynamic_pad=False,
                                    allow_smaller_final_batch=True,
                                    shared_name=None,
                                    name=None
                                )



        return image_batch,label_batch, path_batch
    

    def _matrix_to_list(self,train_split_file,train_split_label,image=False,resample_list=False,
                      resamples_list_fill=False):
        list_train_split_file=[]
        list_train_label=[]
        if resample_list ==False:
            resample_list=[]
            for class_ in range(len(self.classes)):
                resample_list.append(1)
       
        if resamples_list_fill==False:
            resamples_list_fill=[]
            for class_ in range(len(self.classes)):
                resamples_list_fill.append(0)
        
        if (len(resample_list) != len(self.classes)) or (len(resamples_list_fill) != len(self.classes)):
            raise NameError('length of resample_list or resamples_list_fill out of class number' )
    
                            
        
        
        for class_ in range(len(train_split_file)):
                class_data_path=train_split_file[class_]
                class_label=train_split_label[class_]
                for up_times in range(resample_list[class_]):
                    for index in range(len(class_data_path)):
                        
                        list_train_split_file.append(class_data_path[index])
                        list_train_label.append(class_label[index])
                
                ##this is for samples fill after multiplication
                for index in range(len(class_data_path)):
                    if resamples_list_fill[class_] == index:
                        break
                    
                    list_train_split_file.append(class_data_path[index])
                    list_train_label.append(class_label[index])
        
        list_train_split_file=np.asarray(list_train_split_file)
        list_train_label=np.asarray(list_train_label)
        if image == True:

            gpu_options = tf.GPUOptions(allow_growth = True)
            with tf.Graph().as_default(),tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as tmp_sess:
                list_train_split_file_tensor,list_train_label_tensor,path_tensor=self._tf_train_image_reader(list_train_split_file,list_train_label,self.image_size
                                                                                                   ,num_threads=self.num_threads)
                tmp_coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess=tmp_sess,coord=tmp_coord)
                list_train_split_file,list_train_label,path=tmp_sess.run([list_train_split_file_tensor,list_train_label_tensor,path_tensor])

                tmp_sess.close()
            
            self.validation_data_path =  path
            list_train_split_file=np.asarray(list_train_split_file)
            list_train_label=np.asarray(list_train_label)
           
        return list_train_split_file,list_train_label

    def _train_test_split(self,split=False):
        train_file=[]
        train_label=[]
        train_split_file=[]
        train_split_label=[]

        
        for class_ in range(len(self.storage_file)):
            storage_file=self.storage_file[class_]
            storage_file=np.asarray(storage_file)
            
            storage_label=self.storage_label[class_]
            storage_label=np.asarray(storage_label)
            
            storage_file_length=len(storage_file)
            
            shuffle_index = np.arange(0,storage_file_length)
            
            np.random.shuffle(shuffle_index)
            
            
            
            
            
            storage_file=storage_file[shuffle_index]
            storage_label=storage_label[shuffle_index]
            
            if type(split) is float:
                train_file.append(storage_file[:int((1-split)*(storage_file_length))])
                train_split_file.append(storage_file[int((1-split)*(storage_file_length)):])

                train_label.append(storage_label[:int((1-split)*(storage_file_length))])
                train_split_label.append(storage_label[int((1-split)*(storage_file_length)):])

            elif type(split) is int:
                if split<storage_file_length:
                    train_file.append(storage_file[:(storage_file_length-split)])
                    train_split_file.append(storage_file[(storage_file_length-split):])

                    train_label.append(storage_label[:(storage_file_length-split)])
                    train_split_label.append(storage_label[(storage_file_length-split):])
                elif split>=storage_file_length:
                    raise NameError('split('+str(storage_file_length)+') is large than testing set!') 

        train_file=np.asarray(train_file)
        train_split_file=np.asarray(train_split_file)

        train_label=np.asarray(train_label)
        train_split_label=np.asarray(train_split_label)
        
        self.filepath_of_validation=train_split_file
        
        
        train_split_file,train_split_label=self._matrix_to_list(train_split_file,
                                                              train_split_label,image=True)

        return train_file,train_label,train_split_file,train_split_label
    
    def _sample_balance(self):
        max_samples=self.batch_table[self.batch_table.argmax()]
        resamples_list=[]
        resamples_list_fill=[]
        for number_of_sample in self.batch_table:

            resamples=int((max_samples)/number_of_sample)
            remainder=int(max_samples%number_of_sample)
            resamples_list.append(resamples)
            resamples_list_fill.append(remainder)
        return resamples_list,resamples_list_fill

    
    def _convert_to_tf_batch(self,image,label,batch_size,image_size=(299,299),num_threads=4):
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.float32)
        input_queue = tf.train.slice_input_producer([image, label], shuffle=True)
        label = input_queue[1]
        image_path=input_queue[0]

        image_contents = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        shape=image.get_shape()
       
        image=tf.image.resize_images(image,image_size,method=self._resize_method)

        if self.preprocessing_setting != False:
            image=self._image_preprocesing(image,self.preprocessing_setting)
            image.set_shape([image_size[0],image_size[1],3])
   
        if self.custom_function!=False:

            image=tf.py_func(self.custom_function,[image],tf.float32)
            image.set_shape([image_size[0],image_size[1],3])

        capacity =  5 * batch_size 

        image_batch, label_batch, path_batch = tf.train.batch([image, label,image_path],
                                    batch_size=batch_size,
                                    num_threads=num_threads,
                                    capacity=capacity,
                                    enqueue_many=False,
                                    shapes=None,
                                    dynamic_pad=False,
                                    allow_smaller_final_batch=False,
                                    shared_name=None,
                                    name=None
                                )


        image_batch = tf.cast(image_batch, tf.float32)
        return image_batch,label_batch,path_batch
     

    def _tf_train_batch(self,image,label,batch_size,image_size=(299,299),num_threads=4):
      
 
        
        
        filelength=len(image)
        self.epoch=int(filelength/batch_size-0.1)+1
       
        
        if self._cpu == False:
            image_batch,label_batch,path_batch=self._convert_to_tf_batch(image,label,batch_size,
                                                              image_size=image_size,num_threads=num_threads)
        elif self._cpu == True:
            with tf.device('/cpu:0'):
                image_batch,label_batch,path_batch=self._convert_to_tf_batch(image,label,batch_size,
                                                              image_size=image_size,num_threads=num_threads)
        
        return image_batch,label_batch,path_batch
    
    def _convert_tf_batch_to_numpy_batch(self,tf_batch):
        while True:
            train_batch=self.sess.run(tf_batch)
            for index in range(len(train_batch)):
                train_batch[index]=np.asarray(train_batch[index])
      
            yield train_batch
    
    def _image_preprocesing_numpy(self,image):  
        cval= 'constant'
        fill_mode = 0
        cval=self.preprocessing_setting['cval']
        fill_mode=self.preprocessing_setting['fill_mode']
      
        if 'zoom_range' in self.preprocessing_setting:
            image=preprocessing.image.random_zoom(image,self.preprocessing_setting['zoom_range'],
                                                  col_axis=0,row_axis=1,channel_axis=2,
                                                  fill_mode=fill_mode,cval=cval)
         
        if 'rotation_range' in self.preprocessing_setting:
            rotation_range=self.preprocessing_setting['rotation_range']
            if rotation_range <= 360:
                
                image=preprocessing.image.random_rotation(image,rotation_range,col_axis=0,row_axis=1,
                                                              channel_axis=2,fill_mode=fill_mode,cval=cval)
                    
        if ('width_shift_range' in self.preprocessing_setting) or ('height_shift_range' in self.preprocessing_setting):
            
            shift_range=[0,0] #[h,w]
            if 'width_shift_range' in self.preprocessing_setting:
                shift_range[1]=self.preprocessing_setting['width_shift_range']

            if 'height_shift_range' in self.preprocessing_setting:
                shift_range[0]=self.preprocessing_setting['height_shift_range']

            image=preprocessing.image.random_shift(image,shift_range[1],shift_range[0],row_axis=1,col_axis=0,
                                                   channel_axis=2,
                                                   fill_mode=fill_mode,cval=cval)

                
        return image

    def _image_preprocesing(self,image,preprocessing_setting):
        
        if 'crop_or_pad' in preprocessing_setting:
            image=tf.image.resize_image_with_crop_or_pad(image,preprocessing_setting['crop_or_pad'][0],
                                                         preprocessing_setting['crop_or_pad'][1])
        if 'vertical_flip' in preprocessing_setting:
            if preprocessing_setting['vertical_flip']:
                image=tf.image.random_flip_up_down(image)
         

        if 'horizontal_flip' in preprocessing_setting:
            if preprocessing_setting['horizontal_flip']:
                image=tf.image.random_flip_left_right(image)
        
        if 'random_crop' in preprocessing_setting:
            image=tf.random_crop(image,preprocessing_setting['random_crop'])
             

        image=tf.py_func(func=self._image_preprocesing_numpy,inp=[image],Tout=tf.uint8)
        return image

    
    def _get_tf_batch(self,tf_train,batch_size=100,batch_type=True,resamples_list=False,resamples_list_fill=False,numpy=True):
        if batch_type == 'balance':
            resamples_list,resamples_list_fill=self._sample_balance()
            file_lists,labels=self._matrix_to_list(self.storage_file,self.storage_label,image=False,
            resample_list=resamples_list,resamples_list_fill=resamples_list_fill)
        elif batch_type == 'normal':
            file_lists,labels=self._matrix_to_list(self.storage_file,self.storage_label,image=False)
        elif batch_type == 'custom':
            file_lists,labels=self._matrix_to_list(self.storage_file,self.storage_label,image=False,
            resample_list=resamples_list,resamples_list_fill=resamples_list_fill)                 
        else:
            raise NameError('TF batch_data_generater batch_type only support "balance","custom","normal"')
        
        image_batch,label_batch,path_batch=self._tf_train_batch(file_lists,labels,batch_size,self.image_size,self.num_threads)
        
        if self._return_path == True:
            tf_batch=[image_batch,label_batch,path_batch]
        else:
            tf_batch=[image_batch,label_batch]
     
        
        if tf_train=='np':
           
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,allow_growth = True)
            self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=self.sess,coord=coord)
            self._init_tf_Session=True

            tf_batch=self._convert_tf_batch_to_numpy_batch(tf_batch)         

        
        return tf_batch

   
    
    def get_all(self,filepath):
        self._filepath_Initialization(filepath)
        self.storage_file=self._address_collection(self._train_path,type='train')
        self.storage_label=self._label_Initialization(self.storage_file)
        
        test_data=[]
        test_label=[]
        
        for class_ in range(len(self.storage_file)):
            class_data_path=self.storage_file[class_]
            class_label=self.storage_label[class_]
            
            for index in range(len(class_data_path)):
               
                test_data.append(class_data_path[index])
                
                test_label.append(class_label[index])
                
                
        test_data=np.asarray(test_data)
        test_label=np.asarray(test_label)
        self.all_files_path=test_data

        

        gpu_options = tf.GPUOptions(allow_growth = True)
        with tf.Graph().as_default(),tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as tmp_sess:
            test_data,test_label,path_tensor=self._tf_train_image_reader(test_data,test_label,self.image_size,num_threads=self.num_threads)
            tmp_coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=tmp_sess,coord=tmp_coord)
            test_data,test_label,path=tmp_sess.run([test_data,test_label,path_tensor])

            tmp_sess.close()
        
        
        self.testing_data_path = path
        
        
   
        return test_data,test_label   
    

    def get_batch(self,filepath,batch_size=100,batch_type='balance'
                  ,split=False,preprocessing_setting=False,tf_train=True,
                  custom_function=False,resamples_list=False,return_path=False):
        #
        
        self._return_path=return_path
        self._filepath_Initialization(filepath)
        self.storage_file=self._address_collection(self._train_path,type='train')
        self.storage_label=self._label_Initialization(self.storage_file)
        
        
        self.preprocessing_setting=preprocessing_setting
        
        if custom_function is not False:
            self.custom_function=custom_function
            
        
        if split!=False:
            self.split=True
            self.storage_file,self.storage_label,self._train_split_file,self._train_split_label=self._train_test_split(split)
            self.validation_set=[self._train_split_file,self._train_split_label]
        else:
            self.split=False
        

        self.batch_table=self._batch_table_Initialization()
  
        tf_batch=self._get_tf_batch(tf_train,batch_size=batch_size,batch_type=batch_type,resamples_list=resamples_list,resamples_list_fill=False)
                              
            
        return tf_batch