'''
@author Vineeth_Bhaskara
'''
# For Python 2
# Tested on Keras 2.0.8 on Python 2

from sklearn.metrics import roc_auc_score, log_loss
from keras.callbacks import Callback
from keras.utils.data_utils import Sequence


class ModelCheckpointAUC(keras.callbacks.Callback):
    '''
    Save the model at the filepath specified. Pass val_data as a tuple of numpy data and labels, val_data=(val_x,val_y).
    For FP/TP sensitive problems, we would like to Save/Monitor by the AUC score to save the model.
    '''
    
    def __init__(self, val_data, filepath=None, save_model=False, train_data=None, batch_size=256, logfile='./keras_log.log'):
        self.logfile=logfile
        self.batch_size=batch_size
        self.validation_data = DataGen(val_data[0], val_data[1], batch_size=self.batch_size, shuffle=False,
                      onehot_y=True)
        self.true_y = val_data[1]
        
        if train_data is not None:
            self.true_y_train = train_data[1]
            self.train_data = DataGen(train_data[0], train_data[1], batch_size=self.batch_size, shuffle=False,
                      onehot_y=True)
        self.save_model=False
        if save_model:
            self.filepath = filepath
            self.save_model = True
        
        with open(self.logfile, 'a') as f:
            f.write('\n\n===================================\n')
            f.write('TR_LOSS\tVAL_LOSS\tTR_AUC\tVAL_AUC\n')
        
    
    def on_train_begin(self, logs={}):
        self.aucs = [] # validation auc history
        self.losses = []
        self.vallloss = []
        self.max_auc = -1 # absurd init value
        
        self.trainlloss = []
        self.trainaucs = []

    def on_train_end(self, logs={}):
        print 'Validation AUC History: ', self.aucs
        print 'Validation Loss History: ', self.vallloss
        print 'Train AUC History: ', self.trainaucs
        print 'Train Loss History: ', self.trainlloss
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        
        y_pred = self.model.predict_generator(self.validation_data, len(self.validation_data))
        auc_now_val = roc_auc_score(self.true_y, y_pred)
        lloss_now_val = log_loss(self.true_y, y_pred)
        self.aucs.append(auc_now_val)
        self.vallloss.append(lloss_now_val)
        
        y_pred_train = self.model.predict_generator(self.train_data, len(self.train_data))
        auc_now_tr = roc_auc_score(self.true_y_train, y_pred_train)
        lloss_now_tr = log_loss(self.true_y_train, y_pred_train)
        self.trainaucs.append(auc_now_tr)
        self.trainlloss.append(lloss_now_tr)
        
        with open(self.logfile, 'a') as f:
            # 'TR_LOSS\tVAL_LOSS\tTR_AUC\tVAL_AUC\n'
            f.write('{}\t{}\t{}\t{}\n'.format(lloss_now_tr, lloss_now_val, auc_now_tr, auc_now_val))
            
        print '\nEpoch Metrics: train_auc: {}, train_loss: {}, val_auc: {}, val_loss: {}'.format(str(round(auc_now_tr,4)), str(round(lloss_now_tr,4)), str(round(auc_now_val,4)), str(round(lloss_now_val,4))),'\n'
        
        if self.save_model:
            if self.max_auc < auc_now_val:
                print('Saving model to '+self.filepath+' as better val auc.')
                self.model.save(self.filepath, overwrite=True)
                self.max_auc = auc_now_val
                
                # also note the metric down
                with open(self.filepath+'.metrics.txt', 'w') as f:
                    f.write('VAL AUC: ' + str(auc_now_val) + '\n')
                    f.write('VAL Loss: ' + str(lloss_now_val) + '\n')
                    f.write('Train AUC: ' + str(auc_now_tr) + '\n')
                    f.write('Train Loss: ' + str(lloss_now_tr) + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
    
class DataGen(Sequence):
    '''
    Just pass in numpy arrays of data and there you go! You get a generator that will return batches of data
    w/wo shuffling. You may add more transformations here if you need.
    '''

    def __init__(self, x_set, y_set, batch_size, shuffle=True, onehot_y=True, seed=28081994):
        np.random.seed(seed)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        if shuffle:
            tot = np.arange(self.x.shape[0])
            np.random.shuffle(tot)
            self.x = self.x[tot, :]
            if onehot_y:
                self.y = self.y[tot, :]
            else:
                self.y = self.y[tot]
        

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
