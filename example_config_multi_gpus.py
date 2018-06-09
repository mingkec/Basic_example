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

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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
        

    
    return [net],custom_setting   


def custom_loss(system_setting,custom_setting):
    training=system_setting['Training']
    if training == True:
        if 'Optimizer' not in custom_setting:
            custom_setting['Optimizer'] = tf.train.AdamOptimizer(system_setting['learning_rate_tensor'],epsilon=0.001)
        if 'gradients' not in custom_setting:
            custom_setting['gradients'] = []
        if 'batch_normal_update_ops' not in custom_setting:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
            custom_setting['batch_normal_update_ops'] = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    op = custom_setting['Optimizer']
    
    
    
    training=system_setting['Training']
    logits=system_setting['network'][0]
    labels=system_setting['y_label']
    
    loss =tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
    
    if training == True:
        gradients = op.compute_gradients(loss)
        custom_setting['gradients'].append(gradients)
        
    return [loss],custom_setting


def train_op(system_setting,custom_setting):
    
    lr=system_setting['learning_rate_tensor']
    total_loss=system_setting['total_loss'][0]
    op = custom_setting['Optimizer']
    global_step = tf.Variable(0,name='global_step',trainable=False)

    with tf.control_dependencies(custom_setting['batch_normal_update_ops']):
        gradients = average_gradients(custom_setting['gradients'])
        train_op = op.apply_gradients(gradients)
        return [train_op],custom_setting


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
system_setting['BoardTitles']='example'
system_setting['model_name']='example_config_multi_gpus_sum_gradient'
system_setting['model']=model_layer
system_setting['multi_GPU_calculation'] = 'syn_avg'
system_setting['loss_function']=custom_loss   
system_setting['optimizer_function']=train_op  
system_setting['num_gpu']=2
system_setting['accuracy_func']=accuracy_func
system_setting['max_validation_batch']=24
system_setting['allow_soft_placement']=True
system_setting['model_metrics']=[metrics_function]
system_setting['max_to_keep_ckpt'] = 5