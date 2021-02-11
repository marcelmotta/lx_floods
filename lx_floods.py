
# In[1]: Import modules and setup working directory

import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from scipy.stats import t, ttest_ind

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['figure.dpi'] = 300 # DEFAULT = 72

print("Default working directory: '{}'".format(os.getcwd()))
os.chdir(r'C:\Users\Marcel Motta\OneDrive - NOVAIMS\lx_floods')
print("New working directory: '{}'".format(os.getcwd()))

# In[2]: Create methods

def getIncidents(path_xls, event_code):
    """ Import reported incidents files, remove incomplete years, remove duplicates """
    df = pd.read_excel(path_xls, dtype=object)
    df['datetime'] = df['Data'].dt.floor('h')
    df = df.loc[(df.Ano != 2011) & (df.Ano != 2012)]
    df.drop_duplicates(subset='ID', inplace=True)
    df.rename(columns={'ID': 'Event_ID', 
                       'Data': 'datetime_actual',
                       'Estacoes_clima_lx_ID': 'station'}, 
              inplace=True)
    df.sort_values(by='datetime_actual', inplace=True)
    df.set_index(keys=['datetime', 'station'], inplace=True)
    
    if event_code.empty == False:
        df = df.loc[df['Event_type'].isin(event_code)]
    df2 = df[['target']]
    
    return df2
   
def getWeather(path_txt):
    """ Import weather files, retrieve and format dates, remove data anomalies """
    df = pd.read_fwf(path_txt)
    df = df.dropna(how='all')
    df['date'] =  \
        df['ANO'].astype(int).astype(str) + '-' + \
        df['MS'].astype(int).astype(str) + '-' + \
        df['DI'].astype(int).astype(str)
    df['time'] = \
        df['HR'].astype(int).astype(str) + ':00:00'
    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'], format='%Y/%m/%d %H:%M:%S')

    df.loc[(df['01200535'] == -990), '01200535'] = np.nan
    df.loc[(df['01200579'] == -990), '01200579'] = np.nan
    df.loc[(df['01210762'] == -990), '01210762'] = np.nan
    df['avg'] = df[['01200535', '01200579', '01210762']].mean(axis=1)
    df.drop(['ANO','MS','DI','HR','date','time'], axis=1, inplace=True)
    df.set_index(keys='datetime', inplace=True)
    
    return df

def getStations(station_id):
    """ Merge weather files by station, fix wind direction range """
    df = pd.concat([temp[station_id],
                    temp['avg'],
                    hum[station_id],
                    hum['avg'],
                    precip[station_id],
                    precip['avg'],
                    sun[station_id],
                    sun['avg'],
                    wind_speed[station_id],
                    wind_speed['avg'],
                    wind_dir[station_id]],
        axis=1, 
        keys=['temp', 'temp_avg', 'hum', 'hum_avg' ,'precip', 'precip_avg', 
              'sun', 'sun_avg', 'wind_speed', 'wind_speed_avg', 'wind_dir'])

    df['wind_dir'].replace({360: 0}, inplace=True)
    df['station'] = station_id

    return df

def imputeDistance(target_station, closest_station, furthest_station):
    """ Impute missing values using nearby station """
    target_station.fillna(closest_station, inplace=True)
    target_station.fillna(furthest_station, inplace=True)
        
    return None

def imputeTime(target_station, closest_station, furthest_station):
    """ Impute missing values using nearby observations in the time-series """
    target_station.fillna(method='ffill', limit=1, inplace=True)
    target_station.fillna(method='bfill', limit=1, inplace=True)
    
    closest_station.fillna(method='ffill', limit=1, inplace=True)
    closest_station.fillna(method='bfill', limit=1, inplace=True)
    
    furthest_station.fillna(method='ffill', limit=1, inplace=True)
    furthest_station.fillna(method='bfill', limit=1, inplace=True)
    
    return None

def imputeKNN(target_station):
    """ Impute data using k-nearest neighbors"""
    independentVariables = ['temp', 'temp_avg', 'hum', 'hum_avg', 
                            'precip', 'precip_avg', 'sun', 'sun_avg', 
                            'wind_speed', 'wind_speed_avg']
    targetVar = 'wind_dir'
    
    X_train = target_station[independentVariables]
    y_train = target_station[targetVar]
    
    model = KNNImputer(n_neighbors=2, weights='uniform', copy=False)
    model.fit_transform(X_train)

    target_station = pd.concat([X_train, 
                                y_train, 
                                target_station['station']], axis=1)  
    
    return target_station

def imputeRF(target_station):
    """ Impute missing values using a Random Forest model """
    independentVariables = ['station', 'temp', 'temp_avg', 'hum', 'hum_avg', 
                            'precip', 'precip_avg', 'sun', 'sun_avg', 
                            'wind_speed', 'wind_speed_avg']
    targetVar = 'wind_dir'

    train_set = target_station.dropna(how='any')

    X_train, X_test, y_train, y_test = train_test_split(train_set[independentVariables],
                                                        train_set[targetVar],
                                                        test_size=0.25,
                                                        random_state=42)
    
    model = RandomForestRegressor(n_jobs=-1, n_estimators=100, random_state=42) 
    model.fit(X_train, y_train)
        
    missing_wind = target_station.dropna(subset=independentVariables, how='any')
    missing_wind = missing_wind.loc[missing_wind[targetVar].isna()]    
    missing_wind = missing_wind.assign(new_wind = \
        model.predict(X = missing_wind[independentVariables]))
    
    target_station['wind_dir_imp'] = missing_wind['new_wind']
    target_station['wind_dir'] = target_station[['wind_dir_imp', 'wind_dir']].sum(axis=1)
    
    target_station.drop('wind_dir_imp', axis=1, inplace=True)
    #train_score = model.score(X_train, y_train)
    #test_score = model.score(X_test, y_test)
    
    #print('{} \n R2 score for target station: {} \n R2 score for closest station: {}' \
    #      .format(model.get_params(), train_score, test_score)) 
    
    return None

def createProxies(df):
    """ Create proxies for weekday, season, period and wind diretion, then encode categories """                          
    df['datetime'] = df.index
    
    week_style=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    df['weekday'] = df['datetime'].dt.day_name()
    df['weekday'] = pd.Categorical(df['weekday'], 
                                   ordered=True, 
                                   categories=week_style)
    
    weekday_enc = pd.get_dummies(df['weekday'], prefix='weekday')
    df = pd.concat([df, weekday_enc], axis=1)
    
    periods = ['Late Night (0h-4h)', 'Early Morning (4h-8h)', 'Morning (8h-12h)',
               'Noon (12h-16h)', 'Evening (16h-20h)', 'Night (20h-0h)']
    df['period'] = (df['datetime'].dt.hour % 24 + 4) // 4
    df['period'].replace({1: periods[0],
                          2: periods[1],
                          3: periods[2],
                          4: periods[3],
                          5: periods[4],
                          6: periods[5]}, inplace=True)
    df['period'] = pd.Categorical(df['period'], ordered=True, categories=periods)
    
    period_enc = pd.get_dummies(df['period'], prefix='period')
    df = pd.concat([df, period_enc], axis=1)
    
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    df['season'] = (df['datetime'].dt.month % 12 + 3) // 3
    df['season'].replace({1: seasons[0],
                          2: seasons[1], 
                          3: seasons[2], 
                          4: seasons[3]}, inplace=True)
    df['season'] = pd.Categorical(df['season'], ordered=True, categories=seasons)
    
    season_enc = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_enc], axis=1)
    
    df['year'] = df['datetime'].dt.year
    df.drop('datetime', axis=1, inplace=True)
        
    try:
        df['wind_cardinal'] = ((df['wind_dir'] - 22.5) % 360 + 45) // 45
        df['wind_cardinal'].replace({1: 'North-East',
                                     2: 'East',
                                     3: 'South-East',
                                     4: 'South',
                                     5: 'South-West',
                                     6: 'West',
                                     7: 'North-West',
                                     8: 'North'}, inplace=True)
        df['wind_cardinal'] = df['wind_cardinal'].astype('category')
        wind_enc = pd.get_dummies(df['wind_cardinal'], prefix='wind')
        df = pd.concat([df, wind_enc], axis=1)
        df.drop(['wind_dir'], axis=1, inplace=True)
        
    except Exception:
        pass

    df.set_index('station', append=True, inplace=True)
    
    return df

def cumWeather(df, window):
    """ Create moving averages for weather measurements, given a window size """
    temp_cum = []
    hum_cum = []
    precip_cum = []
    sun_cum = []
    wind_speed_cum = []
    for i in range(len(df)):
        if i < window:
            temp_cum.append(np.mean(df['temp'][0:i+1]))
            hum_cum.append(np.mean(df['hum'][0:i+1]))
            precip_cum.append(np.mean(df['precip'][0:i+1]))
            sun_cum.append(np.mean(df['sun'][0:i+1]))
            wind_speed_cum.append(np.mean(df['wind_speed'][0:i+1]))
        else:
            temp_cum.append(np.mean(df['temp'][i-window+1:i+1]))
            hum_cum.append(np.mean(df['hum'][i-window+1:i+1]))
            precip_cum.append(np.mean(df['precip'][i-window+1:i+1]))
            sun_cum.append(np.mean(df['sun'][i-window+1:i+1]))
            wind_speed_cum.append(np.mean(df['wind_speed'][i-window+1:i+1]))
    df2 = pd.concat([pd.Series(temp_cum),
                     pd.Series(hum_cum),
                     pd.Series(precip_cum),
                     pd.Series(sun_cum),
                     pd.Series(wind_speed_cum)],
        axis=1, 
        keys=['temp_{}h'.format(window),
              'hum_{}h'.format(window),
              'precip_{}h'.format(window),
              'sun_{}h'.format(window), 
              'wind_speed_{}h'.format(window)])
    df2.set_index(df.index, inplace=True)
    df3 = pd.concat([df, df2], axis=1)
    df4 =  pd.concat([df3['station'],
                      df3.filter(regex='^temp', axis=1), 
                      df3.filter(regex='^hum', axis=1),
                      df3.filter(regex='^precip', axis=1),
                      df3.filter(regex='^sun', axis=1),
                      df3.filter(regex='^wind_speed', axis=1),
                      df3.filter(regex='^wind_dir', axis=1)], 
        axis=1)
                         
    return df4

def cumWeather_k(var):
    """ Find best window for cumulative weather measures """
    
    global gago_coutinho, geofisico, ajuda, data
    gago_ = gago_coutinho.copy()
    geo_ = geofisico.copy()
    ajuda_= ajuda.copy()
    
    window = [2, 5, 10, 20]
    
    # CALCULATE CUMULATIVE VALUES USING WINDOW LIST
    for i in window:
        gago_ = cumWeather(gago_, i)
        geo_ = cumWeather(geo_, i)
        ajuda_ = cumWeather(ajuda_, i)
    
    # AGGREGATE WITH COMPLETE DATA SET
    data_gago = gago_.reset_index().merge(data, on=['datetime', 'station'], \
                                          how='left', sort='datetime')
    data_gago['target'].fillna(0, inplace=True)
    data_gago['station'] = 'gago_coutinho'
    
    data_geo = geo_.reset_index().merge(data, on=['datetime', 'station'], \
                                        how='left', sort='datetime')
    data_geo['target'].fillna(0, inplace=True)
    data_geo['station'] = 'geofisico'
    
    data_ajuda = ajuda_.reset_index().merge(data, on=['datetime', 'station'], \
                                            how='left', sort='datetime')
    data_ajuda['target'].fillna(0, inplace=True)
    data_ajuda['station'] = 'ajuda'
    
    data_all = pd.concat([data_gago, data_geo, data_ajuda], axis=0) # TO BE USED FOR PLOTS
    data_all = pd.concat([data_all, pd.get_dummies(data_all['station'], prefix='station')], axis=1)
    data_all = data_all.loc[(data_all['precip_avg'] > 0) | (data_all['precip_2h'] > 0)]
    
    X = data_all.filter(regex=r'^(?=.*{})(?!.*avg).*'.format(var), axis=1)
    y = data_all['target']

    # GET FEATURE VALUE
    feat_value = selectFeatures(X, 
                                y, 
                                RandomForestClassifier(random_state=42), 
                                scale=None)
    
    # PLOT RESULTS       
    fig, axes = plt.subplots(figsize=(10,6))
    axes.barh(feat_value['variable'],
              feat_value['importance'],
              facecolor='slategrey',
              height=0.75,
              alpha=0.5)
    
    axes.xaxis.grid(True, color='#dddddd', zorder=0)
    axes.set_axisbelow(True)
    axes.spines['top'].set_color('white')
    axes.spines['left'].set_color('white')
    axes.spines['bottom'].set_color('white')
    axes.spines['right'].set_color('white')
    
    plt.show()
    
    return feat_value

def getPCA(df, pc, scale_in, scale_out):
    """ Standardize data and get corresponding PCAs """
    
    cont = df.select_dtypes('float64', 'int64').columns.to_list()
    df_cont = df[cont]
    
    if scale_in == 1:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cont)
        df2 = pd.DataFrame(df_scaled, index=df_cont.index, columns=df_cont.columns)
    else:
        df2 = df_cont.copy()

    pca = PCA(n_components=pc, random_state=42)
    df3 = pd.DataFrame(pca.fit_transform(df2), index=df_cont.index)
    df3.set_axis(df3.columns.values + 1, axis=1, inplace=True)
    df3 = df3.add_prefix('PC_')

    if scale_out == 1:
        scaler = MinMaxScaler()
        df3_scaled = scaler.fit_transform(df3)
        df3 = pd.DataFrame(df3_scaled, index=df3.index, columns=df3.columns)    
       
    eigenvec = pd.DataFrame(pca.components_, columns=df_cont.columns)
    var = pd.DataFrame(pca.explained_variance_ratio_, columns=['PCA_Variance_Ratio'])
    df5 = pd.concat([eigenvec, var], axis=1)
    df5.set_axis(df5.index.values + 1, axis=0, inplace=True)
    df5.set_index('PC_' + df5.index.astype(str), inplace=True)
        
    print('Number of components: {}'.format(len(pca.components_)))
    print('Total Explained Variance: {:.2f}'.format(sum(pca.explained_variance_ratio_)))
    print('PC Explained Variance: {}'.format(pca.explained_variance_ratio_))
    return df3, df5

def getCorr(df, coef='pearson', plot=1, filter_dup=True):
    """ Calculate correlation between two variables and output paired results"""
    df2 = df.corr(method=coef)
    size_x = max(12, round((len(df2) + 1)/6) * 6) - 2
    size_y = max(10, round((len(df2) + 1)/6) * 5) - 2

    if plot == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(size_x, size_y))
        sns.heatmap(df2.round(2), 
                    annot=True, 
                    cmap='RdBu',
                    vmin=-1,
                    vmax=1,
                    linewidth=0.5)       
        plt.show()

    df3 = df2.stack().reset_index()
    
    if filter_dup == True:
        df4 = np.column_stack([np.sort(df3.iloc[:, [0,1]].values, axis=1),
                               df3.iloc[:, 2]])
        df5 = pd.DataFrame(df4).infer_objects()
    else:
        df5 = df3.copy()
        
    df5.set_axis(['var_1', 'var_2', 'corr'], axis=1, inplace=True)
    df5['corr_abs'] = np.abs(df5['corr'])
    df6 = df5.drop_duplicates()
    df7 = df6.loc[df6['var_1'] != df6['var_2']]
    df8 = df7.sort_values(by='corr_abs', ascending=False)
    
    return df8.reset_index(drop=True)

def rescale(df_X, df_y, method, seed=42):
    """ Partition samples and apply feature scaling to continuous variables """
    if isinstance(df_X, pd.Series):
        df_X = df_X.to_frame()
        
    X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                        df_y,
                                                        test_size=0.25,
                                                        random_state=seed)
    
    cont = df_X.select_dtypes(include=['float64', 'int64']).columns.to_list()
    disc = df_X.columns.difference(cont).to_list()
    try:
        scaler = method.fit(X_train[cont])
        train_out = scaler.transform(X_train[cont])
        train_out_df = pd.DataFrame(train_out, index=X_train.index, columns=cont)
        X_train_r = pd.concat([train_out_df, X_train[disc]], axis=1)
        
        test_out = scaler.transform(X_test[cont])
        test_out_df = pd.DataFrame(test_out, index=X_test.index, columns=cont)
        X_test_r = pd.concat([test_out_df, X_test[disc]], axis=1)
    except (AttributeError, TypeError) as err:
        print('Scaling method "{}" not valid, returned "{}"' \
              .format(type(method).__name__, err.__class__.__name__))
        X_train_r, X_test_r = X_train, X_test

    return X_train_r, X_test_r, y_train, y_test

def selectFeatures(df_X, df_y, model, scale):
    """ Measure dependence between independent and dependent variables, list best features """
    X_train, X_test, y_train, y_test = rescale(df_X, df_y, method=scale)

    model_ = model.fit(X_train, y_train)
    
    try:
        feat_value = model_.feature_importances_
    except AttributeError:
        try:
            feat_value = model_.coef_
        except:
            print('Model "{}" not valid'.format(type(model).__name__))
            return None
    
    chi2_ = GenericUnivariateSelect(chi2, mode='k_best').fit(X_train, y_train)
    ftest_ = GenericUnivariateSelect(f_classif, mode='k_best').fit(X_train, y_train)
    rfe_ = RFECV(model,
                 cv=StratifiedKFold(10, random_state=42, shuffle=True),
                 scoring=make_scorer(matthews_corrcoef),
                 n_jobs=-1).fit(X_train, y_train)

    features = pd.DataFrame({'variable': df_X.columns, 
                             'chi2': np.round(chi2_.pvalues_, 2),
                             'ftest': np.round(ftest_.pvalues_, 2),
                             'rfe': rfe_.ranking_,
                             'importance': np.round(feat_value, 2)})

    train_score = rfe_.score(X_train, y_train)
    test_score = rfe_.score(X_test, y_test)
    
    print('Model: {} \n'
          'R2 score for training sample: {:.2f} \n'
          'R2 score for test sample: {:.2f}' \
              .format(type(model).__name__, 
                      train_score, 
                      test_score))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    axes.set_xlabel('Number of features selected')
    axes.set_ylabel('Cross validation score \n of number of selected features')
    axes.plot(np.arange(1, len(rfe_.grid_scores_) + 1), 
              rfe_.grid_scores_, 
              label='_nolegend_')
    axes.axvline(np.argmax(rfe_.grid_scores_) + 1, 
                 color='slategray', 
                 linestyle='--')
    
    axes.legend(labels=['Optimal feature set = {}' \
                        .format(np.argmax(rfe_.grid_scores_) + 1)], loc='best')
    axes.set_ylim(0.5, 1)
    axes.invert_xaxis()
    axes.yaxis.grid(True, color='#dddddd', zorder=0)
    axes.set_axisbelow(True)
    axes.spines['top'].set_color('white')
    axes.spines['left'].set_color('white')
    axes.spines['bottom'].set_color('white')
    axes.spines['right'].set_color('white')
    
    plt.show()
    
    return features.sort_values(['rfe', 'importance'], ascending=True)

def testSampling(df_X, df_y, model, scale, seed=42):
    """ Test sampling methods and ratio using a given set of metrics """      
    X_train, X_test, y_train, y_test = rescale(df_X, df_y, method=scale)

    results = []
    methods = ['none', RandomUnderSampler(), RandomOverSampler(), SMOTEENN()]
                    
    for i in methods:
        ratios = np.linspace(0.1, 1, 10)

        for j in ratios:           
            if i == 'none':
                print('\nMethod: {} Sampling ratio: {:.0%}'.format(i, j))
                X_res, y_res = X_train, y_train
            else:
                i.random_state = seed
                i.sampling_strategy = j
                print('\nMethod: {} Sampling ratio: {:.0%}'.format(type(i).__name__,
                                                                   j))
                X_res, y_res = i.fit_resample(X_train, y_train)

            model_ = model.fit(X_res, y_res)
            y_pred = model_.predict(X_test)
            
            try: 
                y_score = model_.decision_function(X_test)
            except AttributeError:
                y_score = model_.predict_proba(X_test)[:,1]
        
            acc = np.round(accuracy_score(y_test, y_pred), 2)
            auc = np.round(roc_auc_score(y_test, y_score), 2)
            rec = np.round(recall_score(y_test, y_pred), 2)
            f1 = np.round(f1_score(y_test, y_pred), 2)
            mcc = np.round(matthews_corrcoef(y_test, y_pred), 2)
            
            results.append([type(model).__name__, acc, auc, rec, f1, mcc, i, j])
            
            print('Model: {} Accuracy: {:.2f} AUC: {:.2f} Recall: {:.2f} F1: {:.2f} MCC: {:.2f}'  \
                  .format(type(model).__name__, acc, auc, rec, f1, mcc))
        
        results_df = pd.DataFrame(results, columns=['Model', 
                                                    'Accuracy', 
                                                    'AUC', 
                                                    'Recall', 
                                                    'F1', 
                                                    'MCC', 
                                                    'Method',
                                                    'Ratio'])
        results_df.set_index(keys='Model', inplace=True)
   
    for metric in results_df.columns[0:5]:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
        axes.plot(np.linspace(0.1,1,10), 
                  results_df.loc[results_df['Method'] == 'none', metric], 
                  label='none',
                  linestyle='--',
                  color='gray',
                  zorder=1)
                
        for l in methods[1:]:
            axes.plot(np.linspace(0.1,1,10), 
                      results_df.loc[results_df['Method'] == l, metric], 
                      label=type(l).__name__,
                      marker='o',
                      zorder=0)
        
        axes.legend(loc='best')    
        #axes.set_title('Resampling performance for model "{}"'.format(type(model).__name__))
        axes.set_xlabel('Sampling Ratio')
        axes.set_ylabel('{} ({})'.format(metric, type(model).__name__))
        
        axes.set_ylim(top=1)
        axes.xaxis.grid(True, color='#dddddd', zorder=0)
        axes.set_axisbelow(True)
        axes.spines['top'].set_color('white')
        axes.spines['left'].set_color('white')
        axes.spines['bottom'].set_color('white')
        axes.spines['right'].set_color('white')
        
        plt.show()

    return results_df

def trainModels(df_X, df_y, scale, seed=42):
    """ Train arbitrary models (w/o CV), return overall performance measures """      
    X_train, X_test, y_train, y_test = rescale(df_X, df_y, method=scale)

    results = []
    models = []
    roc = []
    
    logit = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
    svc = SVC(random_state=seed)
    nb = GaussianNB()
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    knn = KNeighborsClassifier(n_jobs=-1)
    mlp = MLPClassifier(max_iter=1000, random_state=seed)
    
    models.append([type(logit).__name__, logit])
    models.append([type(svc).__name__, svc])
    models.append([type(nb).__name__, nb])
    models.append([type(rf).__name__, rf])
    models.append([type(knn).__name__, knn])
    models.append([type(mlp).__name__, mlp])
    
    for name, model in models:
        model_ = model.fit(X_train, y_train)
        y_pred = model_.predict(X_test)
        
        try: 
            y_score = model_.predict_proba(X_test)[:,1]
        except AttributeError:
            y_score = model_.decision_function(X_test)
    
        fpr, tpr, _ = roc_curve(y_test, y_score)
        acc = np.round(accuracy_score(y_test, y_pred), 2)
        auc = np.round(roc_auc_score(y_test, y_score), 2)
        rec = np.round(recall_score(y_test, y_pred), 2)
        f1 = np.round(f1_score(y_test, y_pred), 2)
        mcc = np.round(matthews_corrcoef(y_test, y_pred), 2)
        
        roc.append([name, auc, fpr, tpr])
        results.append([name, acc, auc, rec, f1, mcc])
        
        print('Model: {} Accuracy: {:.2f} AUC: {:.2f} Recall: {:.2f} F1: {:.2f} MCC: {:.2f}'  \
              .format(name, acc, auc, rec, f1, mcc))
    
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'AUC', 'Recall', 'F1', 'MCC'])
    results_df.set_index(keys='Model', inplace=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    
    for name, auc, fpr_, tpr_ in roc:
        axes[0].plot(fpr_, tpr_, label='{} (AUC = {:.2f})'.format(name, auc))
        axes[1].plot(fpr_, tpr_, label='{} (AUC = {:.2f})'.format(name, auc))

    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    axes[0].legend(loc='best')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    #axes[0].set_title('ROC curve')
    
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    axes[1].legend(loc='best')
    axes[1].set_xlim(0, 0.3)
    axes[1].set_ylim(0.7, 1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    #axes[1].set_title('ROC curve (detail)')
    
    plt.show()

    return results_df

def validateModels(df_X, df_y, scale, scoring, folds, seed=42):
    """ Fit selection of non-optimized models (with CV) for a given performance measure """
    X_train, X_test, y_train, y_test = rescale(df_X, df_y, method=scale)
    
    results = []
    models = []
    names = []

    logit = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
    svc = SVC(random_state=seed)
    nb = GaussianNB()
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    knn = KNeighborsClassifier(n_jobs=-1)
    mlp = MLPClassifier(max_iter=1000, random_state=seed)

    models.append([type(logit).__name__, logit])
    models.append([type(svc).__name__, svc])
    models.append([type(nb).__name__, nb])
    models.append([type(rf).__name__, rf])
    models.append([type(knn).__name__, knn])
    models.append([type(mlp).__name__, mlp])
  
    scoring_ = make_scorer(eval(scoring))

    for name, model in models:
        skfold = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=skfold, scoring=scoring_, n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        print('Model: {} {}: {:.2f} (+/- {:.2f})' \
              .format(name, scoring, cv_results.mean(), cv_results.std()))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
 
    axes.boxplot(
        results,
        whis=[5,95],
        showfliers=True,
        vert=True,
        patch_artist=True,
        widths=0.25,
        boxprops=dict(linestyle='-', linewidth='1'),
        whiskerprops=dict(color='gray', linestyle='-', linewidth='1'),
        capprops=dict(color='gray', linestyle='-', linewidth='1'),
        flierprops=dict(marker='o', markersize='4', markerfacecolor='white', markeredgecolor='silver'),
        medianprops=dict(color='white', alpha=0.5)
    )    
    #axes.set_title('K-fold validation performance (k={})'.format(folds))
    axes.set_xticklabels(names, rotation=45, ha='right')
    axes.set_ylabel('{} (k={})'.format(scoring, folds))
    
    axes.yaxis.grid(True, color='#dddddd', zorder=0)
    axes.set_axisbelow(True)
    axes.spines['top'].set_color('white')
    axes.spines['left'].set_color('white')
    axes.spines['bottom'].set_color('white')
    axes.spines['right'].set_color('white')
    
    plt.show()
    
    results_df = pd.DataFrame(results, index=names).transpose()
    
    return results_df

def loopCV(df_X, df_y, scale, scoring, k_folds, n_loops):
    """ Multiple run K-fold cross-validation """
    results = pd.DataFrame()
    
    for i in range(n_loops):
        print('\nRunning iteration# {}/{} with {} folds'.format(i + 1, n_loops, k_folds))
        cv_ = validateModels(df_X, df_y, scale, scoring, k_folds, i)
        results = results.append(cv_)
    
    cl = 0.95
    df = 10

    print('\nResults for {}x{} cross-validation (CI={:.0%}, df={})'.format(n_loops, 
                                                                         k_folds,
                                                                         cl,
                                                                         df))
    for i in results.columns:
        t_int = np.round(t.interval(alpha=cl, df=df, loc=np.mean(results.loc[:,i]), scale=np.std(results.loc[:,i])), 2)
        t_mean = np.round(t.mean(df=df, loc=np.mean(results.loc[:,i]), scale=np.std(results.loc[:,i])), 2)
        t_std = np.round(t.std(df=df, loc=np.mean(results.loc[:,i]), scale=np.std(results.loc[:,i])), 2)
        
        print('Model: {} Mean: {} Error: {} Interval: {}' \
              .format(i, t_mean, t_std, t_int))
        
    return results

def optimizeModel(df_X, df_y, scale, scoring, folds, seed=42):
    """ Optimize model using a parameter grid and cross-validation  """
    X_train, X_test, y_train, y_test = rescale(df_X, df_y, method=scale)

    model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    
    model_params = {'n_estimators': [[int(x) for x in np.linspace(start=100, stop=1000, num=10)]],
                    'criterion': [['gini', 'entropy']],
                    #'max_depth': [[None, *[int(x) for x in np.linspace(10, 100, num=4)]]],
                    #'min_samples_split': [[2, 5, 10]],
                    #'min_samples_leaf': [[1, 2, 4]],
                    #'max_features': [['auto', 'log2', None]],
                    #'bootstrap': [[True, False]]
                    }
    
    score_ = eval(scoring)
    
    skfold = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    model_ = GridSearchCV(model, 
                          ParameterGrid(model_params),
                          scoring=make_scorer(score_),
                          cv=skfold,
                          n_jobs=-1,
                          verbose=1).fit(X_train, y_train)

    baseline = model.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
        
    optimal = model_.best_estimator_
    y_opt = optimal.predict(X_test)
    
    margin = (score_(y_test, y_opt) - score_(y_test, y_base)) / score_(y_test, y_base)
    
    print('Model: {} Score: {} Baseline: {:.2f} Optimized: {:.2f} Improvement: {:.4f}'  \
          .format(type(model).__name__,
                  scoring, 
                  score_(y_test, y_base),
                  score_(y_test, y_opt),
                  margin))

    print('Best parameter set for optimizing {}: {}'.format(scoring, model_.best_params_))
    
    return model_.best_estimator_

# In[3]: Import input files and  filter by event type, impute missing values and create proxies

event_code = pd.Series([3500, 3501, 3502]) # data['Event_type'], check file with event types
data = getIncidents('occurrences_census_weatherstations_near.xlsx', event_code)

temp = getWeather("dados_hora-lisboa/dados_hor_12.txt")
hum = getWeather("dados_hora-lisboa/dados_hor_17.txt")
wind_dir = getWeather("dados_hora-lisboa/dados_hor_36.txt")
wind_speed = getWeather("dados_hora-lisboa/dados_hor_40.txt")
precip = getWeather("dados_hora-lisboa/dados_hor_60.txt")
sun = getWeather("dados_hora-lisboa/dados_hor_62.txt")

# In[4]: Data preparation, imputation, feature engineering

geofisico = getStations('01200535')
ajuda = getStations('01210762')
gago_coutinho = getStations('01200579')

imputeDistance(gago_coutinho, geofisico, ajuda)
imputeDistance(ajuda, geofisico, gago_coutinho)
imputeDistance(geofisico, ajuda, gago_coutinho)
imputeTime(geofisico, ajuda, gago_coutinho)
gago_coutinho = imputeKNN(gago_coutinho)
geofisico = imputeKNN(geofisico)
ajuda = imputeKNN(ajuda)

imputeRF(gago_coutinho)
imputeRF(geofisico)
imputeRF(ajuda)

#k_weather = cumWeather_k('precip')

gago_coutinho = cumWeather(gago_coutinho, 2)
geofisico = cumWeather(geofisico, 2)
ajuda = cumWeather(ajuda, 2)

gago_coutinho = createProxies(gago_coutinho)
geofisico = createProxies(geofisico)
ajuda = createProxies(ajuda)

# In[5]: Merge all weather data into a single table, filter by rainy weather 

data_gago = gago_coutinho.reset_index().merge(data, on=['datetime', 'station'], \
                                              how='left', sort='datetime')
data_gago['target'].fillna(0, inplace=True)
data_gago['station'] = 'gago_coutinho'

data_geo = geofisico.reset_index().merge(data, on=['datetime', 'station'], \
                                         how='left', sort='datetime')
data_geo['target'].fillna(0, inplace=True)
data_geo['station'] = 'geofisico'

data_ajuda = ajuda.reset_index().merge(data, on=['datetime', 'station'], \
                                       how='left', sort='datetime')
data_ajuda['target'].fillna(0, inplace=True)
data_ajuda['station'] = 'ajuda'

data_all = pd.concat([data_gago, data_geo, data_ajuda], axis=0) # TO BE USED FOR PLOTS
data_all = pd.concat([data_all, pd.get_dummies(data_all['station'], prefix='station')], axis=1)
data_all = data_all.loc[(data_all['precip_avg'] > 0) | (data_all['precip_2h'] > 0)]

X = data_all.drop(['datetime', 
                   'weekday', 
                   'period', 
                   'season', 
                   'wind_cardinal',
                   'station',
                   'target'], 
                  axis=1)
y = data_all['target']

# In[6]: Plot correlation matrix

# CORRELATION MATRIX - BEFORE FEATURE ENGINEERING
data_all_weather = pd.concat([data_all['temp'],
                              data_all['hum'],
                              data_all['precip'],
                              data_all['sun'],
                              data_all['wind_speed'], 
                              data_all['target'].astype('int8')], axis=1)

sns.pairplot(data_all_weather, hue='target', palette={0: 'skyblue', 1: 'tomato'})
getCorr(data_all_weather)

# CORRELATION MATRIX - AFTER FEATURE ENGINEERING
_cols = data_all.select_dtypes(include=['float64']).columns.to_list()
_cols.append('target')
data_all_weather_new = data_all[_cols]

getCorr(data_all_weather_new)

# In[7]: Fit, cross-validate and evaluate models

train_out = trainModels(X, 
                        y, 
                        scale=StandardScaler())

cv_out = validateModels(X, 
                        y, 
                        scale=StandardScaler(), 
                        scoring='matthews_corrcoef', 
                        folds=10)

loop_cv = loopCV(X, 
                 y,
                 scale=StandardScaler(), 
                 scoring='matthews_corrcoef', 
                 k_folds=10, 
                 n_loops=10)

#ttest_ind(loop_cv.iloc[:,0], loop_cv.iloc[:,1])

# In[8]: Optimize models and return feature importances

# DROP RENDUNDANT VARIABLES
X = X.drop(['temp',
            'hum',
            'precip',
            'sun',
            'wind_speed'], axis=1)

# DROP LESS INFORMATIVE VARIABLES
feat_value = selectFeatures(X, 
                            y, 
                            RandomForestClassifier(random_state=42), 
                            scale=None)
feat_selected = feat_value.loc[feat_value['rfe'] == 1, 'variable']
X = X[feat_selected]

# TEST SAMPLING METHODS
sampling_out = testSampling(X, 
                            y, 
                            model=RandomForestClassifier(random_state=42, n_jobs=-1),
                            scale=StandardScaler())

opt_model = optimizeModel(X, 
                          y, 
                          scale=StandardScaler(), 
                          scoring='matthews_corrcoef', 
                          folds=10)

feat_value_cv = pd.DataFrame({'variable': X.columns, 'score': opt_model.feature_importances_})

# In[9]: Test final model using unseen data from the storm Elsa

def getIncidents_test(path_xls, event_code):
    df = pd.read_excel(path_xls)
    df['datetime'] = df['datetime'].dt.floor('h')
    df['station'] = '0' + df['station'].astype(str)
    df.drop_duplicates(subset='code_rsb', inplace=True)

    if event_code.empty == False:
        df = df.loc[df['typology_code'].isin(event_code)]
    df['target'] = 1
    
    df2 = df[['datetime', 'station', 'target']]
    df2.set_index(keys=['datetime', 'station'], inplace=True)
    return df2

def getWeather_test(path_txt):
    """ Import weather files, retrieve and format dates, remove data anomalies """
    df = pd.read_fwf(path_txt)
    df = df.dropna(how='all')
    
    df['date'] =  \
        df['ANO'].astype(int).astype(str) + '-' + \
        df['MS'].astype(int).astype(str) + '-' + \
        df['DI'].astype(int).astype(str)
    df['time'] = \
        df['HR'].astype(int).astype(str) + ':00:00'
    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'], format='%Y/%m/%d %H:%M:%S')

    df['station'] = "0" + df['ESTACAO'].astype(str)
    df.drop(['ANO','MS','DI','HR','date','time', 'ESTACAO', 'P_M_MD', 'DD_FFX', 'FF_MAX'],
            axis=1,
            inplace=True)
    df.rename(columns={'T_MED': 'temp',
                       'HR_MED': 'hum',
                       'DD_MED': 'wind_dir',
                       'FF_MED': 'wind_speed',
                       'PR_QTD': 'precip',
                       'RG_TOT': 'sun'}, 
              inplace=True)
    
    df.replace(-990, np.nan, inplace=True)
    df.replace(990, np.nan, inplace=True)
    df['wind_dir'].replace(360, 0, inplace=True)
    
    df2 = df[['datetime', 'station', 'temp', 'hum', 'wind_dir', \
             'wind_speed', 'precip', 'sun']]

    df2.set_index(keys=['datetime'], inplace=True)
    
    return df2

# IMPORT FLOOD RECORDS
data_test = getIncidents_test('storm_sample/Ocorrencias_elsa_NEAR_station.xlsx', 
                              event_code)

# IMPORT DATA
geo_test = getWeather_test('storm_sample/01200535_hor.txt')
ajuda_test = getWeather_test('storm_sample/01210762_hor.txt')
gago_test = getWeather_test('storm_sample/01200579_hor.txt')

# GET CITY AVERAGES
data_all_test = pd.concat([geo_test, ajuda_test, gago_test], axis=0).reset_index()
data_all_test = data_all_test.drop(['wind_dir'], axis=1)
data_all_test = data_all_test.groupby(by='datetime').mean()
data_all_test = data_all_test.add_suffix('_avg')

geo_test = pd.concat([geo_test, data_all_test], axis=1)
ajuda_test = pd.concat([ajuda_test, data_all_test], axis=1)
gago_test = pd.concat([gago_test, data_all_test], axis=1)

# IMPUTE MISSING VALUES
imputeDistance(gago_test, geo_test, ajuda_test)
imputeDistance(ajuda_test, geo_test, gago_test)
imputeDistance(geo_test, ajuda_test, gago_test)
imputeTime(gago_test, ajuda_test, geo_test)
gago_test = imputeKNN(gago_test)
geo_test = imputeKNN(geo_test)
ajuda_test = imputeKNN(ajuda_test)

# GET MOVING AVERAGES
gago_test = cumWeather(gago_test, 2)
geo_test = cumWeather(geo_test, 2)
ajuda_test = cumWeather(ajuda_test, 2)

# ENCODE CATEGORICAL VARIABLES AND CREATE PROXIES
gago_test = createProxies(gago_test)
geo_test = createProxies(geo_test)
ajuda_test = createProxies(ajuda_test)

# MERGE WEATHER STATIONS AND FLOOD RECORDS
data_all_test = pd.concat([gago_test, geo_test, ajuda_test], axis=0)
data_all_test = data_all_test.merge(data_test, left_index=True, right_index=True, how='left')
data_all_test['target'].fillna(0, inplace=True)

# GET STATION NAME
data_all_test.reset_index(inplace=True)
data_all_test['station'].replace('01200535', 'geofisico', inplace=True)
data_all_test['station'].replace('01210762', 'ajuda', inplace=True)
data_all_test['station'].replace('01200579', 'gago_coutinho', inplace=True)
data_all_test = pd.concat([data_all_test, pd.get_dummies(data_all_test['station'], prefix='station')], axis=1)

data_all_test = data_all_test.loc[(data_all_test['datetime'] >= '2019-12-16') &
                                  (data_all_test['datetime'] <= '2019-12-21')]

# GET FINAL INPUTS FOR MODELLING
X2 = data_all_test.drop(['datetime',
                        'weekday',
                        'period',
                        'season',
                        'wind_cardinal',
                        'station',
                        'target'],
                        axis=1)
y2 = data_all_test['target']

# DROP REDUNDANT VARIABLES
X2 = X2.drop(['temp',
            'hum',
            'precip',
            'sun',
            'wind_speed'], axis=1)

# DROP LESS INFORMATIVE VARIABLES
X2 = X2[feat_selected]

# PREDICT NEW INSTANCES
data_all_test['pred_target'] = opt_model.predict(X2)
data_all_test['pred_score'] = opt_model.predict_proba(X2)[:,1]
data_all_test['pred_log_score'] = opt_model.predict_log_proba(X2)[:,1]

# OUTPUT RESULTS
data_all_test.to_excel('lx_floods_out.xlsx')

print('Final Model: {} Accuracy: {:.2f} AUC: {:.2f} Recall: {:.2f} F1: {:.2f} MCC: {:.2f}'  \
      .format(type(opt_model).__name__, 
              accuracy_score(data_all_test['target'], data_all_test['pred_target']),
              roc_auc_score(data_all_test['target'], data_all_test['pred_target']),
              recall_score(data_all_test['target'], data_all_test['pred_target']),
              f1_score(data_all_test['target'], data_all_test['pred_target']),
              matthews_corrcoef(data_all_test['target'], data_all_test['pred_target'])
             )
      )
    
# In[10]: Export all weather data (if necessary)

data_all.drop_duplicates().drop(columns='target').to_excel('ipma_2013-2018.xlsx')
data_all_test.drop_duplicates().drop(columns='target').to_excel('ipma_2019q4.xlsx')

