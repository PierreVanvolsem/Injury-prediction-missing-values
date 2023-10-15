import numpy as np
from Data import Data
import sklearn.metrics
import copy
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.neural_network import MLPClassifier
import json
from sklearn.metrics import classification_report
import pickle
class Case():
    def __init__(self, model, data_format="dataframe",data_missing_value="fill",normalization=False,feature_selection="all"):
        self.model = model
        self.data = Data()
        self.data_format = data_format
        self.best_param = None
        self.model_name = None

        self.normalization = normalization
        self.feature_selection = feature_selection
        self.data_missing_value = data_missing_value

    def train(self,**kwargs):
        X_train, X_test, X_val, y_train, y_test, y_val = self.data.get_data(missing_value=self.data_missing_value,format=self.data_format,feature_selection=self.feature_selection,normalization=self.normalization,validation_set=False)
        
        self.trained_model = self.model(**kwargs).fit(X_train,y_train)
        return self.trained_model

    def find_params_with_optuna(self,objective_maker,number_of_trials=1000):
            
        def train_model(model):
            folds = 10
            skf = StratifiedKFold(n_splits=folds,shuffle=True)

            X_train, X_test, X_val, y_train, y_test, y_val = self.data.get_data(missing_value=self.data_missing_value,format=self.data_format,feature_selection=self.feature_selection,normalization=self.normalization,validation_set=False)

            y_train = y_train.to_numpy().ravel()

            originalclass = []
            predictedclass = []
            for train_index, test_index in skf.split(X_train, y_train):
                trained_model = copy.deepcopy(model)
                trained_model.fit(X_train.iloc[train_index], y_train[train_index])
                preds = trained_model.predict(X_train.iloc[test_index])
                pred_labels = np.rint(preds)
                originalclass.extend(y_train[test_index])
                predictedclass.extend(pred_labels)

            return sklearn.metrics.accuracy_score(originalclass,predictedclass)
        
        objective = objective_maker(train_model)

        # save best params to file
        model_name = self.model.__name__
        fill_mode = self.data_missing_value
        feature_selection = self.feature_selection

        study = optuna.create_study(direction="maximize",
            study_name="%s_%s_%s"%(model_name,fill_mode,feature_selection))
        study.optimize(objective, n_trials=number_of_trials,n_jobs = 1)


        self.best_param = study.best_params

        if self.model == MLPClassifier:
            if "hidden_layer_sizes_l2" in self.best_param.keys():
                model_name = "MLPClassifier_2L"
                self.best_param["hidden_layer_sizes"] = (self.best_param["hidden_layer_sizes_l1"],self.best_param["hidden_layer_sizes_l2"])
                # remove the hidden_layer_sizes_l2 key
                del self.best_param['hidden_layer_sizes_l2']
            else:
                model_name = "MLPClassifier_1L"
                self.best_param["hidden_layer_sizes"] = (self.best_param["hidden_layer_sizes_l1"])

            # remove the hidden_layer_sizes_l1 key
            del self.best_param['hidden_layer_sizes_l1']

        self.model_name = model_name
        file_name = "Hyperparameters/best_params_%s_%s_%s.json"%(model_name,fill_mode,feature_selection)
        with open(file_name,"w") as f:
            json.dump(self.best_param,f)

        return study

    def report(self, model=None, validation=True ,model_name=None,write = True):
        X_train, X_test, X_val, y_train, y_test, y_val = self.data.get_data(missing_value=self.data_missing_value,format=self.data_format,feature_selection=self.feature_selection,normalization=self.normalization,validation_set=True)

        # Get different Validations sets (dropped and filed)
        _, _, X_val_dropped, _, _, y_val_dropped = self.data.get_data(missing_value="drop",format=self.data_format,feature_selection=self.feature_selection,normalization=self.normalization,validation_set=True)
        _, _, X_val_filled, _, _, y_val_filled = self.data.get_data(missing_value="fill",format=self.data_format,feature_selection=self.feature_selection,normalization=self.normalization,validation_set=True)

        if self.model_name is None and model_name is not None:
            self.model_name = model_name
        elif self.model_name is None:
            self.model_name = self.model.__name__

        model_name = self.model.__name__
        fill_mode = self.data_missing_value
        feature_selection = self.feature_selection

        if self.model == MLPClassifier:
            print(self.model_name)
            model_name = self.model_name
            
        if model is None and self.best_param is not None:
            model = self.model(**self.best_param).fit(X_train,y_train)
        elif model is None and self.best_param is None:
            # load best params from file
            best_params_file_name = "Hyperparameters/best_params_%s_%s_%s.json"%(model_name,fill_mode,feature_selection)
            with open(best_params_file_name,"r") as f:
                best_param = json.load(f)
            model = self.model(**best_param).fit(X_train,y_train)
        elif model is None:
            model = self.model.fit(X_train,y_train)

        def evaluate(X,y):
            y_pred = np.round(model.predict(X))
            report = classification_report(y, y_pred)
            from sklearn.metrics import confusion_matrix
            cf_matrix = confusion_matrix(y, y_pred)
            return report, cf_matrix

        test_report, test_cf = evaluate(X_test,y_test)
        train_report, train_cf = evaluate(X_train,y_train)

        if validation:
            val_report_dropped, val_dropped_cf = evaluate(X_val_dropped,y_val_dropped)
            val_report_filled, val_filled_cf = evaluate(X_val_filled,y_val_filled)
            if write:
                results_file_name = "Results/%s_%s_%s.txt"%(model_name,fill_mode,feature_selection)
                
                with open(results_file_name, "w") as text_file:
                    text_file.write("Test report: \n %s\n%s \n Test report: \n %s\n%s \n Val dropped report: \n %s\n%s \n Val filled report: \n %s\n%s"%(train_report, train_cf,test_report,test_cf,val_report_dropped,val_dropped_cf,val_report_filled,val_filled_cf))
            return test_report, val_report_dropped, val_report_filled
        else:
            if write:
                results_file_name = "Results/%s_%s_%s.txt"%(model_name,fill_mode,feature_selection)
                with open(results_file_name, "w") as text_file:
                    text_file.write("Test report: \n %s\n%s \n Test report: \n %s\n%s"%(train_report, train_cf,test_report,test_cf))
            return test_report
        
    
    def save_model(self,model, model_path):
        if self.data_format == "dataframe":
            with open(model_path,'wb') as f:
                pickle.dump(model,f)
        else:    
            raise NotImplementedError
        