import numpy as np

def _concatenate_prediction(GPU_predictions,num_of_gpu,single_prediction=True):
    prediction=[]
    netwrok_outputs=len(GPU_predictions[0])
   
    for netwrok in range(netwrok_outputs):
        prediction_single_netwrok=[]
        for num_gpu in range(num_of_gpu):
            prediction_single_netwrok.append(GPU_predictions[num_gpu][netwrok])

        prediction_single_netwrok=np.concatenate(prediction_single_netwrok)
        prediction.append(prediction_single_netwrok)

    return prediction
#concatenate mini batch prediction
def _batch_concatenation(prediction):
    netwrok_outputs = len(prediction[0])
    prediction_concatenation=[]
    for prediction_single_netwrok_index in range(netwrok_outputs):
        prediction_single_netwrok=[single_prediction[prediction_single_netwrok_index] for single_prediction in prediction]
        prediction_single_netwrok=np.asarray(prediction_single_netwrok)
        prediction_single_netwrok=np.concatenate(prediction_single_netwrok)
        prediction_concatenation.append(prediction_single_netwrok)
    prediction=prediction_concatenation
    return prediction

def _input_check(X_data,y_label=False,num_of_gpu=1):
    #check each GPU get at least one input
    data_number=len(X_data)
    if data_number < num_of_gpu:
        fill_number=num_of_gpu-data_number
        samples_for_fill=np.repeat([X_data[0]],[fill_number],axis=0)
        X_data=np.concatenate([X_data,samples_for_fill])

        if y_label != False:
            label_for_fill=np.repeat([y_label[0]],[fill_number],axis=0)
            y_label=np.concatenate([y_label,label_for_fill])

    return X_data,y_label,data_number

def data_formatting(data,inputs_dict):
   
    validation_data_dict={}
    validation_label_dict={}
    
    #Group Validation Data by 'inputs_dict'
    for index,name in enumerate(inputs_dict['train_or_label']):
        if len(data) > index:

            if name == 'inputs':
                validation_data_dict[inputs_dict['data_name'][index]]=data[index]
            elif name == 'label':
                validation_label_dict[inputs_dict['data_name'][index]]=data[index]
 
    return validation_data_dict,validation_label_dict