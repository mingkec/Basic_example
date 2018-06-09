import numpy as np
from tensorflow.contrib import slim
import tensorflow as tf
from model import inception_v4

import sys
import subprocess
import os
if sys.version_info >(3,0):
    current_dir=subprocess.check_output(['pwd']).decode('UTF-8').split('\n')[0]
    log_path=os.path.join(current_dir,'log')
else:
    print('This version Only support Python 3')


system_setting={}

def model_layer(system_setting,custom_setting):

    X_input=system_setting['X_input']
    dropout_keep_prob_tensor=system_setting['dropout_keep_prob_tensor']
    training=system_setting['Training']
    
    arg_scope=inception_v4.inception_v4_arg_scope(use_batch_norm=True)
    with slim.arg_scope(arg_scope): 
        net,end=inception_v4.inception_v4(X_input, num_classes=10, is_training=training,
                 dropout_keep_prob=dropout_keep_prob_tensor,
                 scope='InceptionV4',
                reuse=not training,
                 create_aux_logits=False)
        if training == True:
            if 'unfreeze_variables' not in custom_setting:
                unfreeze_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model/InceptionV4/Logits')[0:2]
                custom_setting['unfreeze_variables'] = unfreeze_variables
                transfer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model/InceptionV4')[:-2]
                custom_setting['transfer_variables'] = transfer_variables

    return [net],custom_setting   


def custom_loss(system_setting,custom_setting):

    training=system_setting['Training']
    logits=system_setting['network'][0]
    labels=system_setting['y_label']
    
    loss_function=tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)

    return [loss_function],custom_setting


def train_op(system_setting,custom_setting):
    
    lr=system_setting['learning_rate_tensor']
    total_loss=system_setting['total_loss'][0]
    unfreeze_variables = custom_setting['unfreeze_variables']
    global_step = tf.Variable(0,name='global_step',trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op=tf.train.AdamOptimizer(lr,epsilon=0.001).minimize(total_loss,global_step=global_step,var_list = unfreeze_variables)
        return [op],custom_setting

def metrics_function(system_setting,custom_setting):
    #Here is no metrics function
    #Make it by yourself
    #logits,pred=system_setting['prediction']
    #labels=system_setting['label']['y_label']
    #BoardValues={'sensitivity':sensitivity,'specificity':specificity,'accuracy':accuracy}
    #system_setting['BoardValues']=BoardValues
    
    logits=system_setting['prediction'][0]
    label=system_setting['label']['y_label']
    accuracy=np.mean(np.equal(label.argmax(1),logits.argmax(1)))
    
    BoardValues={'accuracy':accuracy}
    system_setting['BoardValues']=BoardValues
    return system_setting,custom_setting

system_setting['current_dir']=current_dir
system_setting['logdir']=log_path
system_setting['BoardTitles']='example_transfer_learning'
system_setting['model_name']='example_transfer_learning'
system_setting['model']=model_layer
system_setting['loss_function']=custom_loss   
system_setting['optimizer_function']=train_op  
system_setting['num_gpu']=1
system_setting['max_validation_batch']=24
system_setting['allow_soft_placement']=True
system_setting['model_metrics']=[metrics_function]
system_setting['max_to_keep_ckpt'] = 5