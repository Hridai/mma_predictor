import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statutils import StatUtil
from sklearn import linear_model
import joblib

''' Initial Setup - Raw Data loads/Joins '''
def init_from_raw():
    df_fighters_in = pd.read_excel('fighter_static.xlsx')
    df_fighters2_in = pd.read_excel('fighter_dynamic.xlsx')
    df_bouts = pd.read_excel('bouts_list.xlsx')
    df_fighters_in.merge(df_fighters2_in,on='Link',how='left',suffixes=('','__2'))
    df_fighters_in.columns = df_fighters_in.columns.str.replace('__2', '')
    return df_fighters_in, df_bouts

''' Initial setup - From Excel (Assuming raw setup already run and saved down '''
def init_from_excel():
    return pd.read_excel('full_fighter_detail.xlsx'), pd.read_excel('bout_list.xlsx')

def merge_fighters_bouts(df_f, df_b):
    df_f['Fighter1'] = df_f.apply(lambda x: x['First'] + ' ' + x['Last'], axis=1)
    df_temp = df_b.merge(df_f, on='Fighter1')
    df_f.rename(columns={'Fighter1':'Fighter2'}, inplace=True)
    df = df_temp.merge(df_f, on='Fighter2')
    df.columns = df.columns.str.replace('_x','_F1')
    df.columns = df.columns.str.replace('_y','_F2')
    return df

class UFCModel():
    def __init__(self, df):
        self.df = df
    
    ordered_weightclasses = ['Flyweight', 'Bantamweight', 'Featherweight',
                             'Lightweight', 'Welterweight', 'Middleweight',
                             'Light Heavyweight', 'Heavyweight']

    def _clean_date(self, d, fmt):
        try:
            res = datetime.strptime(d, fmt)
        except:
            return np.nan
        return res
    
    def _encode_wins(self, s):
        if s.lower() == 'win':
            return 1
        elif s.lower() == 'loss':
            return 0
        else:
            return np.nan
    
    def _onehotencode_col(self, df_in, field_name):
        ''' Returns df with the field_name removed and appends the one-hot-encoded '''
        encoded_df = df_in[field_name].values
        encoded_df = encoded_df.reshape(1,-1).transpose()
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(encoded_df)
        encoder.categories_
        encoded_df = encoder.transform(encoded_df).toarray()
        encoded_df = pd.DataFrame( encoded_df )
        n = df_in[field_name].nunique()
        encoded_df.columns = ['{}_{}'.format(field_name, n) for n in range(1, n + 1)]
        df_in = df_in.drop(field_name,axis=1)
        df_in = pd.concat([df_in,encoded_df],axis=1)
        return df_in.dropna() if df_in.shape[1] > 1 else None
    
    # Unique values in df col, removing blanks
    def _colunique(self, df, colname):
        unq_list = df[colname].unique()
        return [x for x in unq_list if x == x]
    
    def _p2f(self, x):
        return float(str(x).strip('%'))/100
    
    def _set_weightclass(self, w):
        if w < 135.0:
            return 'Flyweight'
        elif w < 145.0:
            return 'Bantamweight'
        elif w < 155.0:
            return 'Featherweight'
        elif w < 170.0:
            return 'Lightweight'
        elif w < 185.0:
            return 'Welterweight'
        elif w < 205.0:
            return 'Middleweight'
        elif w < 220.0:
            return 'Light Heavyweight'
        else:
            return 'Heavyweight'
        
    def clean_datatypes(self):
        self.df = self.df.replace('--',np.nan)
        self.df['EventDate'] = self.df.apply(lambda x: self._clean_date(x['EventDate'], '%b. %d, %Y'), axis=1)
        self.df['DOB_F1'] = self.df.apply(lambda x: self._clean_date(x['DOB_F1'], '%b %d, %Y'), axis=1)
        self.df['DOB_F2'] = self.df.apply(lambda x: self._clean_date(x['DOB_F2'], '%b %d, %Y'), axis=1)
        self.df["Reach_F1"] = pd.to_numeric(self.df["Reach_F1"], downcast="float")
        self.df["Height_F1"] = pd.to_numeric(self.df["Height_F1"], downcast="float")
        self.df["Weight_F1"] = pd.to_numeric(self.df["Weight_F1"], downcast="float")
        self.df["Reach_F2"] = pd.to_numeric(self.df["Reach_F2"], downcast="float")
        self.df["Height_F2"] = pd.to_numeric(self.df["Height_F1"], downcast="float")
        self.df["Weight_F2"] = pd.to_numeric(self.df["Weight_F1"], downcast="float")
        self.df['Weightclass_F1'] = self.df.apply(lambda x: self._set_weightclass(x['Weight_F1']), axis=1)
        self.df['Weightclass_F2'] = self.df.apply(lambda x: self._set_weightclass(x['Weight_F2']), axis=1)
    
    def exploratory_analysis(self, df_f):
        ''' Ideas log:
            Try to find whether there is a fighter peak age by division perhaps
            Explore combining the significant values SPLM etc. Some combo of that edge will be key
            Build in a memory of knockouts, liklihood of greater KO..;
        '''
        df_f = df_f.replace('--',np.nan)
        df_f["SLpM"] = pd.to_numeric(df_f["SLpM"], downcast="float")
        df_f["Str.Acc"] = df_f.apply(lambda x: self._p2f(x['Str.Acc']), axis=1)
        df_f["SApM"] = pd.to_numeric(df_f["SApM"], downcast="float")
        df_f["Str.Def"] = df_f.apply(lambda x: self._p2f(x['Str.Def']), axis=1)
        df_f["TD Avg."] = pd.to_numeric(df_f["TD Avg."], downcast="float")
        df_f["TD Acc."] = df_f.apply(lambda x: self._p2f(x['TD Acc.']), axis=1)
        df_f["TD Def."] = df_f.apply(lambda x: self._p2f(x['TD Def.']), axis=1)
        df_f["Sub. Avg"] = pd.to_numeric(df_f["Sub. Avg"], downcast="float")
        df_f["Weight"] = pd.to_numeric(df_f["Weight"], downcast="float")
        df_f['WeightClass'] = df_f.apply(lambda x: self._set_weightclass(x['Weight']), axis=1)
        
        # Reach Distribution By Weight Class
        fig, axs = plt.subplots(2,4,figsize=(20,10))
        j = 0
        k = 0
        for i, f in enumerate(self.ordered_weightclasses):
            df_reach = df_f.loc[df_f['WeightClass'] == f]['Reach']
            j = 0 if i < 4 else 1
            k = i % 4
            sns.distplot(df_reach, kde=False, ax=axs[j, k]).set_title(f)
            axs[j, k].set(ylabel='Reach')
        fig.suptitle('Reach Distribution by Weightclass')
        plt.show()
        
        stats_to_plot = ['SLpM', 'Str.Acc', 'SApM', 'Str.Def',
           'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg']
        for item in stats_to_plot:
            fig, axs = plt.subplots(2,4,figsize=(20,10))
            j = 0
            k = 0
            for i, f in enumerate(self.ordered_weightclasses):
                df_slpm = df_f.loc[df_f['WeightClass'] == f]
                df_slpm = df_slpm.loc[df_slpm[item] != 0][item]
                mean=df_slpm.mean()
                median=df_slpm.median()
                mode=df_slpm.mode()[0]
                j = 0 if i < 4 else 1
                k = i % 4
                sns.distplot(df_slpm, kde=False, ax=axs[j, k]).set_title(f)
                axs[j, k].axvline(mean, color='r', linestyle='--')
                axs[j, k].axvline(median, color='g', linestyle='-')
                axs[j, k].axvline(mode, color='b', linestyle='-')
                axs[j, k].set(ylabel=f'{item}')
            fig.suptitle(f'{item} Distribution by Weightclass (excl any == 0)')
            plt.legend({'Mean':mean,'Median':median,'Mode':mode})
            plt.show()
    
    def plot_reg_scatter(self, X, y, clf, pngsavepath = ''):
        fig, ax = plt.subplots()
        X_plot = np.linspace(150,227,50)
        ax.scatter(X, y, s=25)
        ax.plot(X_plot, clf.predict(X_plot.reshape(-1,1)), color='orange')
        ax.set_xlabel('Height', fontsize=14)
        ax.set_ylabel('Reach', fontsize=14)
        ax.set_title('Height vs Reach Fitting', fontsize=20)
        fig.tight_layout()
        if pngsavepath != '':
            plt.savefig(pngsavepath)
        plt.show()
        return fig
    
    def infer_reach(self, clf, height, reach):
        if pd.isnull(reach):
            if pd.isnull(height):
                return np.nan
            elif height == 0:
                return np.nan
            else:
                return clf.predict(np.array(height).reshape(1,-1))[0][0]
        else:
            return reach
            
    
    def engineer_features(self):
        # Create DF to engineer and test new features
        self.df['F1ResultEncoded'] = self.df.apply(lambda x: self._encode_wins(x['Fighter1Result']), axis=1)
        self.df_eng = self.df[['F1ResultEncoded','F1Strikes',
           'F2Strikes', 'F1TakeDowns', 'F2TakeDowns', 'F1Subs', 'F2Subs', 'F1Pass',
           'F2Pass','DOB_F1', 'SLpM_F1', 'Str.Acc_F1', 'SApM_F1', 'Str.Def_F1',
           'TD Avg._F1', 'TD Acc._F1', 'TD Def._F1', 'Sub. Avg_F1', 'Weight_F1',
           'Height_F2', 'Weight_F2', 'Reach_F2', 'Stance_F1', 'Reach_F1',
           'Stance_F2','DOB_F2', 'SLpM_F2','Str.Acc_F2', 'SApM_F2', 'Str.Def_F2',
           'TD Avg._F2', 'TD Acc._F2','TD Def._F2', 'Sub. Avg_F2','EventDate','Gender_F1',
           'Height_F1']]
        
        self.df_eng = self.df_eng.replace('--',np.nan)
        
        fields_to_float = ['F1Strikes',
           'F2Strikes', 'F1TakeDowns', 'F2TakeDowns', 'F1Subs', 'F2Subs', 'F1Pass',
           'F2Pass', 'SLpM_F1', 'SApM_F1', 
           'Height_F2', 'Weight_F2', 'Reach_F2',
           'SLpM_F2', 'SApM_F2', 'Height_F1',
           'Sub. Avg_F2', 'Weight_F1', 'Reach_F1']
        
        fields_pct_to_float = ['Str.Acc_F1', 'TD Avg._F1', 'TD Acc._F1',
                               'TD Def._F1', 'Sub. Avg_F1', 'Str.Acc_F2',
                               'Str.Def_F2', 'TD Avg._F2', 'TD Acc._F2',
                               'TD Def._F2', 'Str.Def_F1']
        
        for name in fields_to_float:
            self.df_eng[name] = pd.to_numeric(self.df_eng[name], downcast="float")
    
        for name in fields_pct_to_float:
            self.df_eng[name] = self.df_eng.apply(lambda x: self._p2f(x[name]), axis=1)
        
        df_heightreach = pd.DataFrame(self.df_eng[['Height_F1','Reach_F1']]).sort_values('Height_F1').dropna()
        clf = linear_model.LinearRegression()
        clf.fit(df_heightreach['Height_F1'].to_numpy().reshape(-1,1),df_heightreach['Reach_F1'].to_numpy().reshape(-1,1))

        print('DISP FOR SAVING ONLY')
        joblib.dump(clf, 'reach_from_height_clf.pkl')
        
        # StatUtil().plt_line(df_heightreach['Height_F1'],df_heightreach['Reach_F1'],'Height','Reach','Title',None)
        
        # self.plot_reg_scatter(df_heightreach['Height_F1'].to_numpy().reshape(-1,1), df_heightreach['Reach_F1'].to_numpy().reshape(-1,1), clf)
        
        self.df_eng['Reach_F1'] = self.df_eng.apply(lambda x: self.infer_reach(clf, x['Height_F1'], x['Reach_F1']), axis=1)
        self.df_eng['Reach_F2'] = self.df_eng.apply(lambda x: self.infer_reach(clf, x['Height_F2'], x['Reach_F2']), axis=1)
        
        self.df_eng['diff_strikes'] = self.df_eng['F1Strikes'] - self.df_eng['F2Strikes']
        self.df_eng['diff_takedowns'] = self.df_eng['F1TakeDowns'] - self.df_eng['F2TakeDowns']
        self.df_eng['diff_subs'] = self.df_eng['F1Subs'] - self.df_eng['F2Subs']
        self.df_eng['diff_pass'] = self.df_eng['F1Pass'] - self.df_eng['F2Pass']
        self.df_eng['diff_age'] = self.df_eng.apply(lambda x: (x['DOB_F1'] - x['DOB_F2']).days, axis=1)  
        self.df_eng['diff_SLpM'] = self.df_eng['SLpM_F1'] - self.df_eng['SLpM_F2']
        self.df_eng['diff_SApM'] = self.df_eng['SApM_F1'] - self.df_eng['SApM_F2']
        self.df_eng['diff_StrAcc'] = self.df_eng['Str.Acc_F1'] - self.df_eng['Str.Acc_F2']
        self.df_eng['diff_StrDef'] = self.df_eng['Str.Def_F1'] - self.df_eng['Str.Def_F2']
        self.df_eng['diff_TDAvg'] = self.df_eng['TD Avg._F1'] - self.df_eng['TD Avg._F2']
        self.df_eng['diff_TDAcc'] = self.df_eng['TD Acc._F1'] - self.df_eng['TD Acc._F2']
        self.df_eng['diff_TDDef'] = self.df_eng['TD Def._F1'] - self.df_eng['TD Def._F2']
        self.df_eng['diff_SubAvg'] = self.df_eng['Sub. Avg_F1'] - self.df_eng['Sub. Avg_F2']
        self.df_eng['diff_Reach'] = self.df_eng['Reach_F1'] - self.df_eng['Reach_F2']
        self.df_eng['WeightClass'] = self.df_eng.apply(lambda x: self._set_weightclass(x['Weight_F1']), axis=1)
        self.df_eng['age_when_fight'] = self.df_eng.apply(lambda x: (x['EventDate'] - x['DOB_F1']).days, axis=1)
    
        self.df_eng = self.df_eng.loc[self.df_eng['diff_SLpM'] != 0]
        self.df_eng = self.df_eng.loc[self.df_eng['diff_SApM'] != 0]
        self.df_eng = self.df_eng.loc[self.df_eng['diff_StrAcc'] != 0]
        self.df_eng = self.df_eng.loc[self.df_eng['diff_StrDef'] != 0]
        
        features_corr = ['F1ResultEncoded',
                'diff_age','diff_SLpM','diff_SApM','diff_StrAcc','diff_StrDef','diff_TDAvg',
                'diff_TDAcc','diff_TDDef','diff_SubAvg','diff_Reach', 'age_when_fight'
                ]
        
        corr_matrix_weightclass = pd.DataFrame()
        for name in self.ordered_weightclasses:
            sub_df = self.df_eng.loc[self.df_eng['WeightClass'] == name][features_corr].corr()['F1ResultEncoded']
            sub_df.columns = [name]
            if corr_matrix_weightclass.shape[0] == 0:
                corr_matrix_weightclass = sub_df
            else:
                corr_matrix_weightclass = pd.concat([corr_matrix_weightclass, sub_df], axis=1)
        
        corr_matrix_weightclass.columns = self.ordered_weightclasses
        corr_matrix_weightclass = corr_matrix_weightclass.drop(['F1ResultEncoded'],axis=0)
        
        # cols_keep = ['F1ResultEncoded','diff_age', 'diff_SLpM',
        #    'diff_SApM', 'diff_StrAcc', 'diff_StrDef', 'diff_TDAvg', 'diff_TDAcc',
        #    'diff_TDDef', 'WeightClass', 'age_when_fight','SLpM_F1','SApM_F1','Str.Acc_F1',
        #    'Str.Def_F1','TD Acc._F1','TD Def._F1','Reach_F1']
        
        cols_keep = ['F1ResultEncoded','diff_age', 'diff_SLpM',
           'diff_SApM', 'diff_StrAcc', 'diff_StrDef', 'diff_TDAvg', 'diff_TDAcc',
           'diff_TDDef', 'WeightClass', 'age_when_fight','Gender_F1',
           'diff_Reach']
        
        self.df_eng_allcols = self.df_eng
        self.df_eng = self.df_eng[cols_keep]
        self.df_eng = self.df_eng.dropna()
        
    def preprocess_data(self):
        self.df_prep = self.df_eng
        self.df_prep = self._onehotencode_col(self.df_prep, 'WeightClass')
        self.df_prep = self._onehotencode_col(self.df_prep, 'Gender_F1')
        self.df_prep = self.df_prep.reset_index(drop=True)
        self.y = np.array(self.df_prep['F1ResultEncoded'])
        del self.df_prep['F1ResultEncoded']
        print(f'Shape of final prep data: {self.df_prep.shape[0]},{self.df_prep.shape[1]}')
        self.df_prep_cols = self.df_prep.columns
        self.scaler = StandardScaler().fit(self.df_prep)
        self.X = np.array(self.scaler.transform(self.df_prep))
    
    def run(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC, LinearSVC, NuSVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score
        

        bVotingClassifier = True
        bGridSearch = False
        bIndividModel = False
        bSave = False
        # for ensemble_model in [RandomForestClassifier(n_estimators = 500),
                                # AdaBoostClassifier(n_estimators=60, learning_rate=0.35),
                                # GradientBoostingClassifier(n_estimators=150, max_depth=2)]:
        # for ensemble_model in [KNeighborsClassifier(n_estimators = 500), SVC(), LinearSVC(), NuSVC()]:
        for ensemble_model in [GradientBoostingClassifier()]:
            print('----------------------------')
            print(ensemble_model)
            print('++++++++++++++++++++++++++++')
            # 66.34% Random Forest
            # 66.92% AdaBoost
            # 66.16% Gradient Boost
            self.clf = ensemble_model
            if bVotingClassifier:
                clf1 = RandomForestClassifier(n_estimators = 500)
                clf2 = AdaBoostClassifier(n_estimators=60, learning_rate=0.35)
                clf3 = GradientBoostingClassifier(n_estimators=150, max_depth=2)
                self.eclf = VotingClassifier(estimators=[('rf',clf1),('ada',clf2),('gb',clf3)], voting='soft')
                self.eclf.fit(self.X,self.y)
                self.y_pred = self.eclf.predict(self.X)
                accuracy = accuracy_score(self.y, self.y_pred)
                print(f'Accuracy: {accuracy}')
                
                if bSave:
                    joblib.dump(self.eclf, 'ufc_eclf.pkl')
                    print('Ensemble Classifier Objet Saved')
                    joblib.dump(self.scaler, 'standardscaler.pkl')
                    print('StandardScaler Object Saved')
                    ### Sample usage of the load function
                    # # Load the model from the file 
                    # knn_from_joblib = joblib.load('filename.pkl')  
                      
                    # # Use the loaded model to make predictions 
                    # knn_from_joblib.predict(X_test) 
                
            if bGridSearch:
                n_estimators = [100,150,200,250]
                max_depth = [2,3,4]                
                param_grid = [{'n_estimators': n_estimators,
                               'max_depth': max_depth}]
                
                grid_search = GridSearchCV(self.clf, param_grid, cv=5,
                                           scoring='accuracy',
                                           return_train_score=True,
                                           verbose=2, n_jobs=-1)
                grid_search.fit(self.X, self.y)
                print(f'best params:\n{grid_search.best_params_}')
                self.gridsearch_res = grid_search.cv_results_
                print(f'mean: {self.gridsearch_res["mean_test_score"].mean()}')
                print(f'std: {self.gridsearch_res["mean_test_score"].std()}')
            
            if bIndividModel:
                self.clf.fit(self.X, self.y)
                scores = cross_val_score(self.clf, self.X, self.y, scoring='accuracy', cv=10)
                # self.final_scores = np.sqrt(-scores)
                self.final_scores = scores
                print(f'Scores: {self.final_scores}')
                print(f'Mean: {self.final_scores.mean()}')
                print(f'StdDev: {self.final_scores.std()}')
                self.important_features = pd.DataFrame([self.df_prep_cols,self.clf.feature_importances_])
        
if __name__ == '__main__':
    # init_data()
    df_f, df_b= init_from_excel()
    df = merge_fighters_bouts(df_f, df_b)
    
    '''disp'''
    df = pd.read_excel('full_dataset.xlsx')
    
    model = UFCModel(df)
    model.clean_datatypes()
    # model.exploratory_analysis(df_f)
    model.engineer_features()
    model.preprocess_data()
    model.run()
    # model_important_features = model.important_features.T.sort_values(1)
    
    print('Model Complete')
    