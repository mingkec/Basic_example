from os import listdir
import cv2
import numpy as np
import pandas as pd
import os
import copy
from tensorflow.contrib import slim
import tensorflow as tf
from Taffetas.system.system import *
from Taffetas.system.unit import data_formatting,_batch_concatenation,_concatenate_prediction,_input_check
tf.logging.set_verbosity('ERROR')


class service_operator():
    def __init__(self,inputs,system_setting,custom_setting=False):
        if custom_setting == False:
            custom_setting={}
            
        self.custom_setting=custom_setting

        system_setting=system_setting
   
        system_setting=init_basic_setting(system_setting)

  
      
        system_setting=init_checking_point_and_log_dir(system_setting)


        system_setting['inputs']=inputs
        self.system_setting=init_training_parameter(system_setting)
        

        self.system_setting,self.custom_setting=model_declaration(self.system_setting,self.custom_setting,training=True)
       
        
        print('init test')
        self.system_setting,self.custom_setting=model_declaration(self.system_setting,self.custom_setting,training=False)
        
        self.sess=init_Session_and_GPUOptions(self.system_setting)
        
        
        self._writer_master,self._value_feeder,self.system_setting=init_summary_op(self.system_setting,self.sess)
        
        
        self._init_training_object(self.system_setting)
        
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess.run(init_op)

        print('init End')

    def _init_training_object(self,system_setting):
        
        self.train_op=system_setting['train_op']
        self.training_loss=system_setting['training_loss']
        self.dropout_keep_prob_tensor=system_setting['dropout_keep_prob_tensor']
        self.dropout_keep_prob=system_setting['dropout_keep_prob']
        self.learning_rate_tensor=system_setting['learning_rate_tensor']
        self.learning_rate=system_setting['learning_rate']
        self.global_step=0
        self.checkpoints=0
        self.network_output=system_setting['network_output']
 

        
        self._init_default_losses=np.zeros(system_setting['number_of_loss'])
        
        self.Saver=tf.train.Saver(max_to_keep=self.system_setting['max_to_keep_ckpt'])
        
    def _network_value_extraction(self,data_dict,network_output,basic_setting,split_batchs,max_batch):
        prediction=[]
        
        for start_partition in range(split_batchs+1):
            start_index=start_partition*max_batch
            tmp_batch_dict={}
        
            #Split Validation Data tp mini batch
            for index,name in enumerate(data_dict):
                batch_data=data_dict[name][start_index:(start_index+max_batch)]
                batch_data,_,data_number=_input_check(batch_data,num_of_gpu=self.system_setting['num_gpu'])
                tmp_batch_dict[self.system_setting['validation_set_dict'][name]]=batch_data
                
            tmp_batch_dict={**tmp_batch_dict,**basic_setting}       
            prediction_batch=self.sess.run(network_output,feed_dict=tmp_batch_dict)
            prediction_batch=_concatenate_prediction(prediction_batch,num_of_gpu=self.system_setting['num_gpu'])
            
            if data_number <self.system_setting['num_gpu']:
                for index in range(len(prediction_batch)):  
                    prediction_batch[index]=prediction_batch[index][0:data_number]
        
            prediction.append(prediction_batch)
        return prediction

    def predict(self,data,max_batch=20,accuracy=False):
     
        test_data_length=len(data[0])

        split_batchs=int((test_data_length-0.1)/max_batch)

        basic_setting = {self.dropout_keep_prob_tensor:self.dropout_keep_prob,self.learning_rate_tensor:self.learning_rate}

        validation_data_dict,validation_label_dict=data_formatting(data,self.system_setting['inputs_dict'])
        
        prediction=self._network_value_extraction(validation_data_dict,self.network_output,basic_setting,split_batchs,max_batch)
            
        #concatenate mini batch prediction
        prediction=_batch_concatenation(prediction)
      

        if accuracy==True:

            gpu_options = tf.GPUOptions(allow_growth = True)
      
           
            with tf.Graph().as_default(),tf.Session(config=tf.ConfigProto(log_device_placement=True,gpu_options=gpu_options)) as t_sess:
                prediction_concatenation=[]
                self.system_setting['prediction']=prediction
                for index in range(len(prediction)):
                    single_prediction=tf.convert_to_tensor(prediction[index])

                    prediction_concatenation.append(single_prediction)
                #prediction_concatenation=tuple(prediction_concatenation)
                prediction=prediction_concatenation
                self.system_setting['network']=prediction   
                
                for name in validation_label_dict:
                    self.system_setting[name]=tf.convert_to_tensor(validation_label_dict[name])
                result_of_loss_function,_=self.system_setting['loss_function'](self.system_setting,self.custom_setting)
                custom_loss=t_sess.run(result_of_loss_function)
                
                self.system_setting['label']=validation_label_dict
     
            if self.system_setting['accuracy_func'] != False:
                accuracy,self.custom_setting=self.system_setting['accuracy_func'](self.system_setting,self.custom_setting)
                return custom_loss,accuracy
            else:
                return custom_loss,0
        else:

            return prediction
    

    def evaluate_network(self,data,networks,multi_gpu_merge=True,max_batch=20):
        
        test_data_length=len(data[0])
        split_batchs=int((test_data_length-0.1)/max_batch)
        
        basic_setting = {self.dropout_keep_prob_tensor:self.dropout_keep_prob,self.learning_rate_tensor:self.learning_rate}
        
        validation_data_dict,validation_label_dict=data_formatting(data,self.system_setting['inputs_dict'])
        
        prediction=self._network_value_extraction(validation_data_dict,networks,basic_setting,split_batchs,max_batch)
        #concatenate mini batch prediction
        prediction=_batch_concatenation(prediction)
        return prediction
        
    def evaluate_single_node(self,data,networks):
        basic_setting = {self.dropout_keep_prob_tensor:self.dropout_keep_prob,self.learning_rate_tensor:self.learning_rate}
        validation_data_dict,validation_label_dict=data_formatting(data,self.system_setting['inputs_dict'])
        tmp_batch_dict={}
            #Split Validation Data tp mini batch
        for index,name in enumerate(validation_data_dict):
            batch_data=validation_data_dict[name]
            batch_data,_,data_number=_input_check(batch_data,num_of_gpu=self.system_setting['num_gpu'])
            tmp_batch_dict[self.system_setting['validation_set_dict'][name]]=batch_data

        tmp_batch_dict={**tmp_batch_dict,**basic_setting}       
        prediction_batch=self.sess.run(networks,feed_dict=tmp_batch_dict)
        return prediction_batch
    
    def save_checking_point(self):
       
        self.Saver.save(self.sess, self.system_setting['checking_point_path'], global_step=self.checkpoints)
    
    def restore_checking_point(self,checking_point_path):

        self.Saver.restore(self.sess, checking_point_path)
    
    def restore_checking_point_with_variables(self,checking_point_path,variables_to_restore):
       
        
        Saver=tf.train.Saver(variables_to_restore)
        Saver.restore(self.sess, checking_point_path)

        
    def _write_text_report(self,text_report_data):
   
        text_report_df=pd.DataFrame.from_dict({'text_report_data':text_report_data},orient= 'index')
        
        try:
            if not os.path.exists(self.system_setting['text_report_path']):
                text_report_df.to_csv(self.system_setting['text_report_path'],index=False)
            else:
                major_text_report=pd.read_csv(self.system_setting['text_report_path'])
                major_text_report=pd.concat([major_text_report,text_report_df])
                major_text_report.to_csv(self.system_setting['text_report_path'],index=False)

        except Exception as e:
            print('Error at checkpoints '+str(self.checkpoints))
            print(e) 

 
        
    def _write_summary_and_report(self,system_setting,custom_setting):

        report_data=system_setting['basic_info']
       
        if 'BoardValues' in system_setting:

            report_data={**report_data,**system_setting['BoardValues']}
     
        if self.system_setting['init_summary'] == False:
            self._summary_manager=[]
            with tf.variable_scope(system_setting['BoardTitles']) as scope:
                for Name_of_value in report_data:
                    self._summary_manager.append(tf.summary.scalar(Name_of_value, self._value_feeder))

                       
            self.system_setting['init_summary']=True
   

      
        for summary_index,Name_of_value in enumerate(report_data):
         
            summary=self.sess.run(self._summary_manager[summary_index],feed_dict={self._value_feeder:report_data[Name_of_value]})
            self._writer_master.add_summary(summary,self.checkpoints)
        
   
        self._writer_master.flush()
       
        text_report_data={'model_name':self.system_setting['model_name'],
                          'checking_point_path':self.system_setting['checking_point_path'],
                          'checkpoints':self.checkpoints,
                          'global_step':self.global_step}
        
        text_report_data={**text_report_data,**report_data}

        self._write_text_report(text_report_data)
    
    
        
        
    def model_metrics(self,validation_data,batch_size):
     
        custom_loss,accuracy=self.predict(validation_data,max_batch=batch_size,accuracy=True)

        self.system_setting['Final_Validation_Loss']=custom_loss

        Name_of_losses_list=self.system_setting['Name_of_losses']
        Name_of_losses_dict={}


        for name_index in range(len(self.train_loss)):
            if len(self.train_loss) == len(Name_of_losses_list):
                Name_of_losses_dict[('Training_'+Name_of_losses_list[name_index])]=self.train_loss[name_index]
                Name_of_losses_dict[('Validation_'+Name_of_losses_list[name_index])]=custom_loss[name_index]
            elif (len(self.train_loss) > len(Name_of_losses_list)) and len(Name_of_losses_list) ==1:
                Name_of_losses_dict[('Training_'+Name_of_losses_list[0]+'_'+str(name_index))]=self.train_loss[name_index]
                Name_of_losses_dict[('Validation_'+Name_of_losses_list[0]+'_'+str(name_index))]=custom_loss[name_index]


        if self.system_setting['accuracy_func'] !=False:
            basic_info={'Accuracy':accuracy}
        else:
            basic_info={}
        basic_info={**basic_info,**Name_of_losses_dict}
            
            
            
        self.system_setting['basic_info']=basic_info
        if 'model_metrics' in self.system_setting:
            metrics_functions=self.system_setting['model_metrics']
            for metrics_function in metrics_functions:
                self.system_setting,self.custom_setting =metrics_function(self.system_setting,self.custom_setting)
                
        
            
        return self.system_setting,self.custom_setting
    
    def evaluate(self,validation_data,batch_size=20,save=False):

        self.system_setting,self.custom_setting=self.model_metrics(validation_data,batch_size)
      
        if save == True:
            self.checkpoints+=1
            self.save_checking_point()
           
            self._write_summary_and_report(self.system_setting,self.custom_setting)
  
        return self.system_setting['Final_Validation_Loss']

    
    def train(self):

        _,train_loss=self.sess.run(
            [self.train_op,self.training_loss],
            feed_dict={ 
            self.dropout_keep_prob_tensor:self.dropout_keep_prob,
            self.learning_rate_tensor:self.learning_rate})
       
        self.train_loss+=train_loss
       
        return 1
     

    def fit_generator(self,validation_data,steps_per_epoch,validation_steps,epochs=1,
                      learning_rate_decay_dict=False,learning_rate=False,dropout_keep_prob=False,initial_epoch=False):
        
        if learning_rate != False:
            if type(learning_rate) is float:
                self.learning_rate=learning_rate
            else:
                raise ValueError("learning_rate must be float")
                
        if dropout_keep_prob != False:
            if type(dropout_keep_prob) is float:
                if dropout_keep_prob <= 1.0:
                    self.dropout_keep_prob=dropout_keep_prob
                else:
                    raise ValueError("dropout_keep_prob must smaller than or equal to 1.0")
            else:
                raise ValueError("dropout_keep_prob must be float")
         
        if initial_epoch != False:   
            start_epoch = initial_epoch
            self.checkpoints=start_epoch
       
        else:
            start_epoch = 0
 
      
   
        for epoch in range(start_epoch,epochs,1):
        
            self.train_loss = copy.deepcopy(self._init_default_losses)
      
            if learning_rate_decay_dict != False:
                if epoch in learning_rate_decay_dict:
                    self.learning_rate=learning_rate_decay_dict[epoch]
      
                    
            for step in range(steps_per_epoch):
                self.train()
                self.global_step+=1
          
                if self.global_step % validation_steps == 0 :
                    self.train_loss = self.train_loss/steps_per_epoch
                 
                    print ('Current Step is '+str(self.global_step),'Current Epoch is '+str(epoch+1) )
                  
                    _=self.evaluate(validation_data,save=True,batch_size=self.system_setting['max_validation_batch'])