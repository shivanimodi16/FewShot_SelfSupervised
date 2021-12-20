#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from linear_classifier import TrainLinearClassifiersFS
from dataprep import write_pickle


parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    type = str,
                    help='name of dataset to train linear classifier on')
parser.add_argument('--data-root',
                    type=str,
                    help='Path to training files')
parser.add_argument('--use-validation',
                    type=int,
                    help='wether or not to use validation set as part of training. 1 = True, 0 = False')
parser.add_argument('--repeats',
                    default = 5,
                    type=int,
                    help='Number of times to repeat training')
parser.add_argument('--observations-per-class',
                    nargs='+',
                    default = [3,4,8,16,32,64,128])
parser.add_argument('--cv-folds',
                    default = 3,
                    type=int)
parser.add_argument('--output-path',
                   type=str,
                   help = 'the output directory where results are going to be stored')


def get_idx_to_classes(dataset): 
    if dataset == 'CIFAR10':
        idx_to_classes={ 0: 'airplane',
                         1: 'automobile',
                         2: 'bird',
                         3: 'cat',
                         4: 'deer',
                         5: 'dog',
                         6: 'frog',
                         7: 'horse',
                         8: 'ship',
                         9: 'truck'}

    elif dataset == 'STL10':
        idx_to_classes = {0: 'airplane',
                          1: 'bird',
                          2: 'car',
                          3: 'cat',
                          4: 'deer',
                          5: 'dog',
                          6: 'horse',
                          7: 'monkey',
                          8: 'ship',
                          9: 'truck'}
    else:
        raise ValueError('dataset unknown')
    return idx_to_classes



def get_data(data_root,val=False):

    X_train = np.load(f'{data_root}/X_train.npy')
    X_test = np.load(f'{data_root}/X_test.npy')

    y_train = np.load(f'{data_root}/y_train.npy')
    y_test = np.load(f'{data_root}/y_test.npy')
    
    if val:
        y_val = np.load(f'{data_root}/y_val.npy')
        X_val = np.load(f'{data_root}/X_val.npy')

        X_train = np.vstack([X_train,X_val])
        y_train = np.vstack([np.expand_dims(y_train,axis=1),np.expand_dims(y_val,axis=1)])


    return X_train,X_test,y_train,y_test



def main():

    args = parser.parse_args()

    args.use_validation = bool(args.use_validation)
    args.observations_per_class = [int(i) for i in args.observations_per_class]
    
    svm = LinearSVC(max_iter=5000,
                    dual=False,
                    class_weight='balanced',
                    verbose=2
                   )

    lr = LogisticRegression(multi_class = 'multinomial',max_iter=5000)


    models = {
             'SVM':
              {'model':svm,
              'param_grid':{'C':list(2**np.arange(-19,-3,1,dtype=float))+list(10**np.arange(-7,2,1,dtype=float))}
              },

              'LR':
              {'model':lr,
               'param_grid':{'C':10**np.linspace(-6,6,40)}
              }
             }


    X_train,X_test,y_train,y_test = get_data(args.data_root,val=args.use_validation)

    idx_to_classes = get_idx_to_classes(args.dataset)

    tlc = TrainLinearClassifiersFS(X_train,
                                     X_test,
                                     y_train,
                                     y_test,
                                     args.cv_folds,
                                     idx_to_classes)

    results = tlc.train_few_shot_models(X_train,
                                       y_train.flatten(),
                                       X_test,
                                       y_test.flatten(),
                                       models,
                                       args.observations_per_class,
                                       repeats = args.repeats)

    results_df_accs,clf_report_df,confusion_matrix_summarized_dict = tlc.summarize_all()
    

    write_pickle(results_df_accs,
                 f'{args.output_path}/results_df_accs.pickle')
    write_pickle(clf_report_df,
                 f'{args.output_path}/clf_report_df.pickle')
    write_pickle(confusion_matrix_summarized_dict,
                 f'{args.output_path}/confusion_matrix_summarized_dict.pickle')
    write_pickle(tlc,
                 f'{args.output_path}/trained_classifier_class.pickle')


if __name__=='__main__':
    main()
    
   
    

