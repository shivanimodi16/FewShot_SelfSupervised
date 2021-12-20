import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, average_precision_score,confusion_matrix,classification_report


def mean_average_precision(y_true,y_preds,classes):
    mAP = 0

    for c in classes:
        pos = c
        y_true_bin,y_preds_bin = (y_true==pos)*1,(y_preds==pos)*1
        mAP+=average_precision_score(y_true_bin,y_preds_bin)

    mAP/=len(classes)
    return mAP

def get_confusion_matrix(y_true,y_pred,labels,normalize = None):
    cf_matrix = confusion_matrix(y_true,y_pred,labels=labels,normalize=normalize)
    cf_matrix = pd.DataFrame(cf_matrix,index = labels,columns = labels)
    cf_matrix = pd.concat({'true class': cf_matrix})
    cf_matrix = pd.concat({'predicted class': cf_matrix},axis=1)
    return cf_matrix

class TrainLinearClassifiersFS:
    def __init__(self,
                 X_train,
                 X_test,
                 y_train,
                 y_test,
                 cv,
                 idx_to_class_name,
                 benchmark_metric = 'acc'):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.idx_to_class_name = idx_to_class_name
        self.ns = []
        self.repeats = 0

        
        self.benchmark_metric = benchmark_metric #Unused right now
        best_model = None
        self.results = {}
        self.class_ids = sorted(list(self.idx_to_class_name.keys()))
        self.class_names = [self.idx_to_class_name[i] for i in self.class_ids]
                
    def train_cv_lin_cls(self,
                           X_train: np.array,
                           y_train: np.array,
                           X_test: np.array,
                           y_test: np.array,
                           model: tuple,
                           param_grid: dict):

        sc = StandardScaler()

        pipe = Pipeline(steps = [('standardize',sc),
                                 model])

        param_grid = {f'{model[0]}__{k}':v for k,v in param_grid.items()}


        search = GridSearchCV(pipe, param_grid, n_jobs=-1,cv = self.cv,verbose=2)
        search.fit(X_train, y_train.flatten())
        print("Best parameter (CV score=%0.4f):" % search.best_score_)
        print(search.best_params_)

        best_estimator = search.best_estimator_

        print('Retraining with full training set with optimal hparams...')
        best_estimator.fit(X_train,y_train)

        y_preds = best_estimator.predict(X_test)

        y_preds_names = np.vectorize(self.idx_to_class_name.get)(y_preds)
        y_test_names = np.vectorize(self.idx_to_class_name.get)(y_test)

        acc = accuracy_score(y_test,y_preds)
        print("Test set accuracy=%0.4f:" % acc)

        mAP = mean_average_precision(y_test,y_preds,self.class_ids)

        clf_report = classification_report(y_test_names,y_preds_names, labels=self.class_names,output_dict = 'dict')
        
        clf_report_df = pd.DataFrame({k:v for k,v in clf_report.items() if k in self.class_names})

        cf_matrix = get_confusion_matrix(y_test_names,y_preds_names,self.class_names,normalize = None)

        return best_estimator,acc,mAP,clf_report_df,clf_report,cf_matrix

    def train_models(self,
                     X_train: np.array,
                     y_train: np.array,
                     X_test: np.array,
                     y_test: np.array,
                     models: dict):
        
        results = {}
        
        for k,v in models.items():

            print(f'Training {k}...')
            results[k] = {}

            model = (k,v['model'])
            param_grid = v['param_grid']

            (results[k]['best_estimator'],
             results[k]['test_acc'],
             results[k]['test_ap'],
             results[k]['clf_report_df'],
             results[k]['clf_report'],
             results[k]['cf_matrix']) = self.train_cv_lin_cls(X_train,
                                                              y_train,
                                                              X_test,
                                                              y_test,
                                                              model,
                                                              param_grid)

        best_model_name = max(results, key=lambda v: results[v].get('test_acc'))
        best_model = results[best_model_name]

        return best_model,results
    
    def sample_n_per_class(self,X,y,n,shuffle = True):
        sampled_ids = []
        classes = np.unique(y)
        for c in classes:
            c_ids = np.where(y.flatten()==c)[0]

            if n==-1:#Use all observations
                sampled_ids.extend(list(c_ids))
            else:
                sampled_ids.extend(list(np.random.choice(c_ids,size=n,replace=False)))

        if shuffle:
            np.random.shuffle(sampled_ids)

        return X[sampled_ids],y[sampled_ids]
    
    def train_few_shot_models(self,
                              X_train: np.array,
                              y_train: np.array,
                              X_test: np.array,
                              y_test: np.array,
                              models: dict,
                              ns: list,
                              repeats: int = 5):
        

        for trial in range(repeats):
            
            self.results[trial] = {}
            
            for n in ns:
                print(f'Training with {n} observations per class...')

                X_train_fs, y_train_fs = self.sample_n_per_class(X_train,y_train,n)

                best_model,results = self.train_models(X_train_fs,
                                                       y_train_fs,
                                                       X_test,
                                                       y_test,
                                                       models)

                self.results[trial][n] = {'best_model':best_model,'results':results}

        self.ns = ns
        self.repeats = repeats

        return self.results

    def summarize_confusion_matrix(self):

        assert (self.results != {}), "linear classifier must be trained first"

        temp = self.results[0][list(self.results[0].keys())[0]]['best_model']['cf_matrix']

        confusion_matrix_summarized_dict = {}

        for n in self.ns:
            
            confusion_matrix_summarized = temp.copy()
            for col in confusion_matrix_summarized.columns:
                confusion_matrix_summarized[col].values[:] = 0
            
            for r in range(self.repeats):
                confusion_matrix_summarized+=self.results[r][n]['best_model']['cf_matrix']
            
            confusion_matrix_summarized/=self.repeats
            confusion_matrix_summarized_dict[n] = confusion_matrix_summarized
        
        return confusion_matrix_summarized_dict

    def summarize_clf_report(self):
        clf_report_df = []

        for n in self.ns:
            for r in range(self.repeats):
                temp = self.results[r][n]['best_model']['clf_report_df'].T.reset_index()
                temp['n'] = n
                clf_report_df.append(temp)

        clf_report_df = pd.concat(clf_report_df)
        clf_report_df.rename(columns={'index':'class'},inplace=True)
        
        return clf_report_df

    def summarize_accs(self):
        accs = []

        for i in self.results.keys():
            accs.append([self.results[i][n]['best_model']['test_acc']*100 for n in self.results[i].keys()])
            
        results_df_accs = pd.DataFrame([j for i in [list(zip(self.ns,i)) for i in accs] for j in i],
                    columns = ['num_labeled_sampled','acc'])

        return results_df_accs

    def summarize_all(self):
        results_df_accs = self.summarize_accs()
        clf_report_df = self.summarize_clf_report()
        confusion_matrix_summarized_dict = self.summarize_confusion_matrix()

        return results_df_accs,clf_report_df,confusion_matrix_summarized_dict
