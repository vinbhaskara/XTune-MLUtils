'''
@author Vineeth_Bhaskara
'''
# For Python 2
# Since this is a FP/TP sensitive problem, we would like to Save/Monitor by the AUC score to save the model

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class ModelCheckpointAUC(keras.callbacks.Callback):
    '''
    Save the model at the filepath specified. Pass val_data as a tuple of numpy data and labels, val_data=(val_x,val_y).
    '''
    
    def __init__(self, filepath, val_data, batch_size=16):
        self.filepath=filepath
        self.batch_size=batch_size
        self.validation_data = val_data
    
    def on_train_begin(self, logs={}):
        self.aucs = [] # validation auc history
        self.losses = []
        self.max_auc = -1 # absurd init value

    def on_train_end(self, logs={}):
        print 'Validation AUC History: ', self.aucs
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        auc_now_val = roc_auc_score(self.validation_data[1], y_pred)
        self.aucs.append(auc_now_val)
        print 'val_auc: {}'.format(str(round(auc_now_val,4))),'\n'
        
        if self.max_auc < auc_now_val:
            print('Saving model to '+self.filepath+' as better val auc.')
            self.model.save(self.filepath, overwrite=True)
            self.max_auc = auc_now_val

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return