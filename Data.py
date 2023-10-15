import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=55, layers=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Autoencoder, self).__init__()
        self.relu = nn.ReLU()
        self.encoder1 = nn.Linear(input_dim*2, hidden_dim).to(device)
        self.list_encoders = []
        self.list_decoders = []
        for layer in range(layers-1):
            self.list_encoders.append(nn.Linear(hidden_dim, hidden_dim).to(device))

        for layer in range(layers-1):
            self.list_decoders.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        
        self.decoder = nn.Linear(hidden_dim, input_dim).to(device)

    def forward(self, x, return_encoding = False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        encoded = self.relu(self.encoder1(x).to(device))
        for encoder in self.list_encoders:
            encoded = self.relu(encoder(encoded).to(device))

        if return_encoding:
            return encoded
        else:
            decoded = encoded
            for decoder in self.list_decoders:
                decoded = self.relu(decoder(decoded).to(device))
            decoded = self.decoder(decoded).to(device)
            return decoded

class DFDataset(Dataset):
    def __init__(self, x_df:pd.DataFrame,y_df:pd.DataFrame,mean = None, std = None, auto_encoder=False):
        self.y_torch = torch.from_numpy(y_df.values)
        self.auto_encoder = auto_encoder
        if mean is None:
            self.mean = x_df.mean()
        else:
            self.mean = mean
        if std is None:
            self.std = x_df.std()
        else:
            self.std = std
        normalized=(x_df-self.mean)/self.std
        self.x_torch = torch.from_numpy(normalized.values)

    def __len__(self):
        return len(self.x_torch)

    def get_mean(self):
        return self.mean.values

    def get_std(self):
        return self.std.values

    def __getitem__(self, idx):
        if self.auto_encoder:
            return self.x_torch[idx], self.x_torch[idx] # no need for labels
        else:
            return self.x_torch[idx], self.y_torch[idx] 

class Data():
    def __init__(self):
        self.seizoen2 = self.load_data("Data.seizoen.2.csv")
        self.seizoen3 = self.load_data("Data.seizoen.3.csv")

        # Only keep first injury
        self.seizoen2 = self.seizoen2[self.seizoen2["INJURY_N"]<=1].reset_index(drop=True)
        self.seizoen3 = self.seizoen3[self.seizoen3["INJURY_N"]<=1].reset_index(drop=True)

        self.seizoen2_prepared , self.seizoen2_encoders = self.data_preparation(self.seizoen2)
        self.seizoen3_prepared , self.seizoen3_encoders = self.data_preparation(self.seizoen3)

        self.seed = 7
        self.test_size = 0.3

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load autoencoder
        input_dim=27
        hidden_dim=55
        layers=1
        epoch=17000
        n_neighbors=26
        self.auto_encoder = Autoencoder(input_dim,hidden_dim,layers).to(self.device)
        self.auto_encoder.load_state_dict(torch.load("denoising model/model_"+str(epoch)+"_final.pt"))
        
        # create KNN imputer
        from sklearn.impute import KNNImputer
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        
    def load_data(self,filename) -> pd.DataFrame:
        df = pd.read_csv(filename, sep=',').iloc[:: , 1::]
        return df
    
    def data_preparation(self,df:pd.DataFrame):
        from sklearn import preprocessing
        # currently encoder overwrites the df
        def label_encode(df,columns):
            # mask to avoid encoding nan values
            mask = df.fillna('NaN')[:] != 'NaN'
            
            encoders = []
            for column in columns:
                le = preprocessing.LabelEncoder()
                df[column] = le.fit_transform(df[column])
                encoders += [le]
            return df[mask] , encoders
    
        df, encoders = label_encode(df,["SEASON.TIMING","TRAINING.MATCH","ONDERGROND","ZIJDE","LICHAAMSDEEL","SOORT.BLESSURE","OVERBELASTING.TRAUMA","RECURRENT.NIEUW","CONTACT","VOORKEURSBEEN","SPELPOSITIE.DICHOTOOM","SCHOOL","ANDERE.SPORT"])
        df["AANTAL.JAREN.VOETBAL"] = df["AANTAL.JAREN.VOETBAL"].replace("nee", 0).astype(np.float64) #replace "nee" values to 0 and convert to float64
        return df, encoders
    
    def drop_missing_values(self,df:pd.DataFrame) -> pd.DataFrame:
        return df.dropna()
    
    def fill_missing_values(self,df:pd.DataFrame)-> pd.DataFrame:
        features = self.feature_selection(df)
        targets = self.target_selection(df)
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=self.test_size, random_state=self.seed)
        features_normalized=(X_train-X_train.mean())/X_train.std()
        imputer = self.imputer.fit(features_normalized)
        return pd.concat([self.fill_na(features,imputer),targets],axis=1)
    
    def get_torch(self,X:pd.DataFrame,y:pd.DataFrame,batch_size=700):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)
        train_data = DFDataset(X_train,y_train)
        test_data = DFDataset(X_test,y_test,X_train.mean(),X_train.std())

        X_val = self.feature_selection(self.seizoen3_prepared_missing_values_handled)
        y_val = self.target_selection(self.seizoen3_prepared_missing_values_handled)
        validation_data = DFDataset(X_val,y_val,X_train.mean(),X_train.std())

        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=validation_data,batch_size=batch_size,shuffle=True)
        return train_loader, test_loader,val_loader
    
    def feature_selection(self,df:pd.DataFrame) -> pd.DataFrame:
        return df[["LEEFTIJD.TESTDAG","LENGTE","GEWICHT","VET.","BMI","ZITHOOGTE","BEENLENGTE","MATURITY.OFFSET","SBJ.MAX","CMJ.MAX","CMJ.ARMEN.MAX","SIT.UP","S.R","JS","MS","BB","KTK3","T.TEST.LINKS","T.TEST.RECHTS","DRIBBELTEST.ZONDER.BAL","DRIBBELTEST.MET.BAL","BESTE.5M","BESTE.10M","BESTE.20M","BESTE.30M","YOYO.AFSTAND","APHV"]] #"VOORKEURSBEEN","SPELPOSITIE.DICHOTOOM","SCHOOL","AANTAL.JAREN.VOETBAL","ANDERE.SPORT"

    def feature_selection_correlation(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Returns the features with the highest point-biserial correlation with the target.
        threshold is 0.2 (rounded)
        """
        return df[["LEEFTIJD.TESTDAG","LENGTE","GEWICHT","BEENLENGTE","MATURITY.OFFSET","CMJ.MAX","CMJ.ARMEN.MAX","BESTE.20M","BESTE.30M"]] #"VOORKEURSBEEN","SPELPOSITIE.DICHOTOOM","SCHOOL","AANTAL.JAREN.VOETBAL","ANDERE.SPORT"


    def target_selection(self,df:pd.DataFrame) -> pd.DataFrame:
        return df[["INJURY_N"]]
    
    def fill_na(self,df:pd.DataFrame,imputer):
        mask = df.isna().astype(int)
        torch_features = torch.tensor(np.concatenate([imputer.transform(df),mask.to_numpy()],axis=1)).float()
        train_data = DFDataset(df,df) # dummy dataset to get mean and std. No y_df is needed 
        encoding = pd.DataFrame(self.denormalise(self.auto_encoder(torch_features.to(self.device).float()),train_data).detach().cpu().numpy())
        encoding.columns = df.columns
        return df.fillna(encoding)
    
    def denormalise(self,normalised_data,dataset):
        denormalised_data = normalised_data[:]*torch.from_numpy(dataset.get_std()).to(self.device)+torch.from_numpy(dataset.get_mean()).to(self.device)
        return denormalised_data
    
    def get_data(self,content="seizoen2",format="dataframe",missing_value="drop",feature_selection="all",normalization=False,validation_set=True):
        """Returns the data in the specified format.
        :param content: "seizoen2" or "seizoen3" or "all" or "autoencoder"
        :param format: "dataframe" or "tensor"
        :param missing_value: "drop" or "fill"
        :param feature_selection: "all" or "correlation"
        :param validation_set: True or False if FALSE the validation set is None
        :return: data in the specified format
        """
        #default values
        feature_selection_function = self.feature_selection
        target_selection_function = self.target_selection
        missing_value_function = self.drop_missing_values 
        data = self.seizoen2_prepared
        format_function = lambda X,y: (train_test_split(X, y, test_size=self.test_size,random_state=self.seed),None, None)
        get_validation_function = lambda x: x

        # content
        if content == "seizeon2":
            data = self.seizoen2_prepared
        elif content == "seizoen3":
            data = self.seizoen3_prepared
        elif content == "all":
            data = pd.concat([self.seizoen2_prepared,self.seizoen3_prepared])
        elif content == "autoencoder":
            data = self.seizoen2_prepared
            missing_value = self.drop_missing_values

        # missing value
        if missing_value == "drop":
            missing_value_function = self.drop_missing_values 
        elif missing_value == "fill":
            missing_value_function = self.fill_missing_values

        # feature selection
        if feature_selection == "correlation":
            feature_selection_function = self.feature_selection_correlation
        elif feature_selection == "all":
            feature_selection_function = self.feature_selection     

        # format
        if format == "tensor":
            # returns train loader and test loader
            format_function = self.get_torch
        elif format == "dataframe":
            # returns X_train, X_test, y_train, y_test in pandas dataframe format
            # X_val and y_val are returned if validation_set is True
            # self.feature_selection always use all the features
            transformed_val_data = missing_value_function(pd.concat([self.feature_selection(self.seizoen3_prepared),target_selection_function(self.seizoen3_prepared)],axis=1))
            X_val = self.feature_selection(transformed_val_data)
            y_val = self.target_selection(transformed_val_data)

            def format_function_dataframe(X,y):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,random_state=self.seed)
                return (X_train, X_test, X_val, y_train, y_test, y_val)
            format_function = format_function_dataframe

        if validation_set:
            if format == "tensor":
                # handle missing values for validation set
                self.seizoen3_prepared_missing_values_handled = missing_value_function(pd.concat([self.feature_selection(self.seizoen3_prepared),target_selection_function(self.seizoen3_prepared)],axis=1))
                # returns train loader, test loader and validation loader
                get_validation_function = lambda loaders: (loaders[0], loaders[1], loaders[2])
            elif format == "dataframe":
                if normalization:
                    # Normalize validation set based on traing mean and std
                    def get_validation_function_normalization(data):
                        x_mean = data[0].mean()
                        x_std = data[0].std()
                        X_val = feature_selection_function(data[2])
                        X_train = (data[0]-x_mean)/x_std
                        X_test = (data[1]-x_mean)/x_std
                        X_val = (X_val-x_mean)/x_std
                        return (X_train,X_test,X_val,data[3],data[4],data[5])
                    get_validation_function = get_validation_function_normalization

                else:
                    get_validation_function = lambda data: (data[0],data[1],feature_selection_function(data[2]),data[3],data[4],data[5])
        else:
            # no validation set will be returned
            # do nothing
            if format == "tensor":
                # returns train loader, test loader
                get_validation_function = lambda loaders: (loaders[0], loaders[1], None)
            elif format == "dataframe":
                if normalization:
                    def get_no_validation_function_normalization(data):
                        x_mean = data[0].mean()
                        x_std = data[0].std()
                        X_val = feature_selection_function(data[2])
                        X_train = (data[0]-x_mean)/x_std
                        X_test = (data[1]-x_mean)/x_std
                        X_val = (X_val-x_mean)/x_std
                        return (X_train,X_test,None,data[3],data[4],None)
                    get_validation_function = get_no_validation_function_normalization
                else:
                    get_validation_function = lambda data: (data[0],data[1],None,data[3],data[4],None)

        # self.feature_selection always use all the features
        transformed_data = missing_value_function(pd.concat([self.feature_selection(data),target_selection_function(data)],axis=1))        
        return get_validation_function(format_function(feature_selection_function(transformed_data),target_selection_function(transformed_data)))      
