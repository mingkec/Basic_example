import os
import tensorflow as tf


def init_training_parameter(system_setting):

    if 'checkpoints' not in system_setting:
        system_setting['checkpoints']=0
    
    if 'dropout_keep_prob' not in system_setting:
        system_setting['dropout_keep_prob']=1.0
    system_setting['dropout_keep_prob_tensor']=tf.placeholder(tf.float32)
    
    if 'learning_rate' not in system_setting:
        system_setting['learning_rate']=0.001
    system_setting['learning_rate_tensor']=tf.placeholder(tf.float32)
    
    #init data set
    if 'inputs_dict' not in system_setting:
        system_setting['inputs_dict']={'data_name':['X_input','y_label'],'train_or_label':['inputs','label']}
            
    training_set_dict={}
    validation_set_dict={}
    default_label_dict={}
    
    for index,name in enumerate(system_setting['inputs_dict']['data_name']):
        
        Data_shape=system_setting['inputs'][index].get_shape().as_list()
        Data_shape[0]=None
        
            
        
        training_set_dict[name]=system_setting['inputs'][index]
        if system_setting['inputs_dict']['train_or_label'][index] == 'inputs':
            validation_set_dict[name]=  tf.placeholder(tf.float32, (Data_shape),name=name)
        
          
 

    system_setting['training_set_dict']=training_set_dict
    system_setting['validation_set_dict']=validation_set_dict
    
    #init losses names
    if 'Name_of_losses' not in system_setting:
        system_setting['Name_of_losses']=['Loss']
    
    
    if 'max_validation_batch' not in system_setting:
        system_setting['max_validation_batch']=20
    
    
    return system_setting



def init_checking_point_and_log_dir(system_setting):
    model_name=system_setting['model_name']
    if 'logdir' not in system_setting:
        raise NameError("Please provide a logdir")
    logdir=system_setting['logdir']
    
    checking_point_and_log=os.path.join(logdir,model_name)
    if not os.path.exists(checking_point_and_log):
        os.mkdir(checking_point_and_log)
    
    checking_point_path=os.path.join(checking_point_and_log,'model')
    if not os.path.exists(checking_point_path):
        os.mkdir(checking_point_path)
        
    checking_point_path=os.path.join(checking_point_path,model_name)
    
    log_path=os.path.join(checking_point_and_log,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    text_report_path=os.path.join(checking_point_and_log,model_name+'.csv')
    
    system_setting['checking_point_path']=checking_point_path
    system_setting['text_report_path']=text_report_path
    system_setting['log_path']=log_path
    
    return system_setting

def init_basic_setting(system_setting):
    if 'model_name' not in system_setting:
        system_setting['model_name'] = 'Custom_Model'
    
    if 'num_gpu' not in system_setting:
        system_setting['num_gpu'] = 1
        
    if 'scope' not in system_setting:
        system_setting['scope'] = 'Model'
    
    if 'multi_GPU_calculation' in system_setting:
        loss_collection_gpu=system_setting['multi_GPU_calculation']
        if (loss_collection_gpu != 'syn_avg') and (loss_collection_gpu != 'syn_total'):
            raise NameError("multi_GPU_calculation must be syn_avg or syn_total")

        system_setting['multi_GPU_calculation']=loss_collection_gpu
    else:
        system_setting['multi_GPU_calculation']='syn_avg'
    
    
    
    if 'allow_soft_placement' not in system_setting:
        system_setting['allow_soft_placement']=False
           
    
    if 'model' not in system_setting:
        raise NameError("Please provide your model")
        
    if 'loss_function' not in system_setting:
        raise NameError("Please provide loss_function")
    
    if 'optimizer_function' not in system_setting:
        raise NameError("Please provide optimizer")
        
    if 'accuracy_func' not in system_setting:
        system_setting['accuracy_func']=False
        
    if 'max_to_keep_ckpt' not in system_setting:
        system_setting['max_to_keep_ckpt']=100
    
    system_setting['model_model_declaration_finished']=False
    
    return system_setting


def model_declaration(system_setting,custom_setting,training=True):
  

        system_setting['Training']=training
        
        if training == True:
 
            model_inputs=system_setting['training_set_dict']
            
        else:
   
            model_inputs=system_setting['validation_set_dict']
        
        
        model=system_setting['model']
        loss_function=system_setting['loss_function']
        optimizer_function=system_setting['optimizer_function']
        
        gpu_split_index=[]
        
        partition=tf.shape(list(model_inputs.values())[0])[0]/system_setting['num_gpu']

        for i in range(system_setting['num_gpu']+1):
            index=tf.multiply(partition,i)
            index=tf.cast(index,tf.int32)
            gpu_split_index.append(index)
        
        nets=[]
        losses=[]
 
        model_model_declaration_finished = not training

        
        for i in range(system_setting['num_gpu']):    
            with tf.device('/gpu:%d'% i):
                with tf.variable_scope(system_setting['scope'],values=model_inputs,reuse = model_model_declaration_finished):
                    for name in model_inputs:
                        system_setting[name]=model_inputs[name][gpu_split_index[i]:gpu_split_index[i+1]]

                    net,custom_setting=model(system_setting,custom_setting)
                    
                    system_setting['network']=net
                                 
                    nets.append(net)
                    if training == True:
                        loss,custom_setting=loss_function(system_setting,custom_setting)
                    
                        losses.append(loss)
                    

                    
                    model_model_declaration_finished = True
                    
        system_setting['number_of_network']= len(nets[0]) 
        

        if training ==True:
            system_setting['number_of_loss']= len(losses[0])
            losses=tf.stack(losses,axis=1)
            total_loss=tf.reduce_sum(losses,axis=1)

            if system_setting['multi_GPU_calculation'] != 'syn_total':

                total_loss=tf.divide(total_loss,1.0*system_setting['num_gpu'])
                
            system_setting['total_loss']=total_loss
            train_op,custom_setting=optimizer_function(system_setting,custom_setting)
            system_setting['train_op']=train_op
            system_setting['training_loss']=total_loss   
        else:
            system_setting['network_output']=nets
            #not use here
            #system_setting['validation_loss']=total_loss  
        return system_setting,custom_setting

def init_Session_and_GPUOptions(system_setting):
    if 'sess' not in system_setting:
        gpu_options = tf.GPUOptions(allow_growth = True)

        if system_setting['allow_soft_placement'] == True: 
            config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
        else:
            config=tf.ConfigProto(gpu_options=gpu_options)
    
        sess=tf.Session(config=config)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess,coord=coord)
        system_setting['Session']=sess
    return sess

def init_summary_op(system_setting,sess):
        
    log_path=system_setting['log_path']
    writer_master = tf.summary.FileWriter(log_path,sess.graph)

    system_setting['init_summary']=False

    value_feeder=tf.placeholder(tf.float32)

    return writer_master,value_feeder,system_setting
