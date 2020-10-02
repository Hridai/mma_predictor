import pandas as pd
import numpy as np
import argparse, sys
from datetime import datetime
import joblib

def _clean_date(d, fmt):
    try:
        res = datetime.strptime(d, fmt)
    except:
        return np.nan
    return res

def _set_weightclass(w):
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

def _get_weightclass(s):
    if s == 'Flyweight':
        return 'WeightClass_125'
    elif s == 'Bantamweight':
        return 'WeightClass_135'
    elif s == 'Featherweight':
        return 'WeightClass_145'
    elif s == 'Lightweight':
        return 'WeightClass_155'
    elif s == 'Welterweight':
        return 'WeightClass_170'
    elif s == 'Middleweight':
        return 'WeightClass_185'
    elif s == 'Light Heavyweight':
        return 'WeightClass_205'
    elif s == 'Heavyweight':
        return 'WeightClass_220'

def _p2f(x):
        return float(str(x).strip('%'))/100

def _infer_reach(clf, height, reach):
        if pd.isnull(reach):
            if pd.isnull(height):
                return np.nan
            elif height == 0:
                return np.nan
            else:
                return clf.predict(np.array(height).reshape(1,-1))[0][0]
        else:
            return reach

def check_stat_exists(df):
    return (df['SLpM'] + df['Str.Acc'] + df['SApM'] + df['Str.Def'] + df['TD Avg'] + df['TD Acc'] + df['Sub. Avg']).sum()
        

def run(f1,f2):
    df = pd.read_excel('fighter_static.xlsx')
    df_f1 = df.loc[(df['First'] == str(f1).split(' ')[0].replace("-"," ")) & \
                   (df['Last'] == str(f1).split(' ')[1].replace("-"," "))]
    df_f2 = df.loc[(df['First'] == str(f2).split(' ')[0].replace("-"," ")) & \
                   (df['Last'] == str(f2).split(' ')[1].replace("-"," "))]
    df_f = pd.concat((df_f1,df_f2),axis=0)
    
    # Tidy types
    df_f= df_f.replace('--',np.nan)
    df_f['DOB'] = df_f.apply(lambda x: _clean_date(x['DOB'], '%b %d, %Y'), axis=1)
    df_f["Reach"] = pd.to_numeric(df_f["Reach"], downcast="float")
    reach_clf = joblib.load('reach_from_height_clf.pkl')
    df_f['Reach'] = df_f.apply(lambda x: _infer_reach(reach_clf, x['Height'], x['Reach']), axis=1)
    df_f["Height"] = pd.to_numeric(df_f["Height"], downcast="float")
    df_f["Weight"] = pd.to_numeric(df_f["Weight"], downcast="float")
    df_f['Weightclass'] = df_f.apply(lambda x: _set_weightclass(x['Weight']), axis=1)
    fields_pct_to_float = ['Str.Acc','Str.Def','TD Acc','TD Def']
    for name in fields_pct_to_float:
        df_f[name] = df_f.apply(lambda x: _p2f(x[name]), axis=1)
    df_f = df_f.reset_index(drop=True)
    
    if check_stat_exists(df_f.loc[0]) == 0:
        sys.exit(f'Stat data totals 0 for {f1}') 
    if check_stat_exists(df_f.loc[1]) == 0:
        sys.exit(f'Stat data totals 0 for {f2}') 
    
    # Engineer Cols
    out_colnames = ['diff_age','diff_SLpM','diff_SApM','diff_StrAcc','diff_StrDef','diff_TDAvg','diff_TDAcc','diff_TDDef','age_when_fight','diff_Reach','WeightClass_125','WeightClass_135','WeightClass_145','WeightClass_155','WeightClass_170','WeightClass_185','WeightClass_205','WeightClass_220','Gender_F','Gender_M']
    vals = []
    vals.append((df_f.at[0,'DOB'] - df_f.at[1,'DOB']).days)
    vals.append(df_f.at[0,'SLpM'] - df_f.at[1,'SLpM'])
    vals.append(df_f.at[0,'SApM'] - df_f.at[1,'SApM'])
    vals.append(df_f.at[0,'Str.Acc'] - df_f.at[1,'Str.Acc'])
    vals.append(df_f.at[0,'Str.Def'] - df_f.at[1,'Str.Def'])
    vals.append(df_f.at[0,'TD Avg'] - df_f.at[1,'TD Avg'])
    vals.append(df_f.at[0,'TD Acc'] - df_f.at[1,'TD Acc'])
    vals.append(df_f.at[0,'TD Def'] - df_f.at[1,'TD Def'])
    vals.append((datetime.today() - df_f.at[0,'DOB']).days)
    vals.append(df_f.at[0,'Reach'] - df_f.at[1,'Reach'])
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)
    vals.append(0)

    df_eng = pd.DataFrame([vals], columns=out_colnames)    
    df_eng.at[0,_get_weightclass(df_f.at[0,'Weightclass'])] = 1
    df_eng.at[0,f'Gender_{args.gender.upper()}'] = 1
    
    scaler = joblib.load('standardscaler.pkl')
    X = np.array(scaler.transform(df_eng))
    
    eclf = joblib.load('ufc_eclf.pkl')
    y = eclf.predict(X)
    proba = pd.DataFrame([['Loss%','Win%'],eclf.predict_proba(X)[0]])
    
    print(f'------------------\n{f1} Outcome: {y}\n{proba}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1','--fighter1',default='')
    parser.add_argument('-f2','--fighter2',default='')
    parser.add_argument('-g','--gender',default='M')
    args, leftovers = parser.parse_known_args()
    
    bManual = True
    
    if bManual:
        f1 = 'Carlos Condit'
        f2 = 'Court McGee'
        print('Manual Run...')
        run(f1,f2)
        run(f2,f1)
        print('---------------------')
        
    else:
        if args.fighter1 == '' or args.fighter2 == '':
            print('You must pass two fighter names!')
            sys.exit('Exiting Script...')
    
        run(args.fighter1, args.fighter2)
    
    # Manual Run
