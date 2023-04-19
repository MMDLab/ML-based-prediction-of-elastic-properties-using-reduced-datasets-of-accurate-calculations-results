import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import tqdm

from pymatgen.core import periodic_table
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

df_elements = pd.read_json('./raw_data/elements.json')
df_VASP_valence = pd.read_csv('./raw_data/Default_PAW_potentials_VASP.csv', sep=';')

def set_dopants_for_pure_elements(df, base='Ti'):
    '''
        Add pure elements as a unit cell formula of all possible combinations
        of base and dopants with concentrations of 0.0
        
        Input: any pandas dataframe, contains columns:
            - ucf: unit cell formula in pymatgen format as a standard python dictionary 
            - dopants: python list of dopants (list of str)
            - base: str
    '''
    all_dopants = []
    for idx, row in df.dropna(axis='rows').iterrows():
        dopants = row['dopants']
        all_dopants.append(dopants)

    for dopants_list in all_dopants:
        pure_element_row = df[df['dopants'].isna()].copy()
        pure_element_row = pure_element_row[pure_element_row['base'] == base]
        pure_element_row['dopants'] = [dopants_list]
        base = pure_element_row['base'].values[0]
        ucf = {base:1.0}
        for dop in pure_element_row['dopants'].values[0]:
            ucf[dop] = 0.0
        pure_element_row['ucf'] = [ucf]
        df = pd.concat([df, pure_element_row], ignore_index=True)
    df = df.drop_duplicates(subset=['ucf'], keep='last')
    return df.dropna(axis='rows').reset_index(drop=True)

def mlb_feats_from_elemental_hull(df):
    '''
        Add features extracted from elemental phase diagrams from materials project
        
        Input: any pandas dataframe, contains column:
            - ucf: unit cell formula in pymatgen format as a standard python dictionary 
    '''
    hull_feats = pd.read_json('./raw_data/hull_feats.json')
    
    tmp_df = df.copy()

    mlb = MultiLabelBinarizer()    
    for feat in tqdm.tqdm(hull_feats.columns):
        if feat != 'element':
            elements = df['ucf'].apply(lambda x: list(x.keys()))
            elements_labels = pd.DataFrame(mlb.fit_transform(elements),
                               columns=[f'{i}_{feat}' for i in mlb.classes_],
                               index=elements.index)
            tmp_df = pd.concat([tmp_df, elements_labels], axis=1)
        
            columns_to_set = [f'{i}_{feat}' for i in mlb.classes_]
            for idx in tmp_df.index.to_list():
                data = tmp_df.loc[[idx]]
                for feat_name in columns_to_set:
                    if data[feat_name].values[0] != 0:
                        symbol = feat_name.replace(f'_{feat}','')
                        feat_value = hull_feats[hull_feats['element']==symbol][feat].values[0]
                        tmp_df.loc[idx, feat_name] = feat_value
    
    for feat in tqdm.tqdm(hull_feats.columns):
        if feat != 'element':
            for idx in tmp_df.index.to_list():
                avg_prop = 0
                for symbol, concentraion in tmp_df.loc[idx, 'ucf'].items():
                    ffilter = hull_feats['element'] == symbol
                    prop_value = hull_feats[ffilter][feat].values[0]
                    avg_prop += prop_value*concentraion
                tmp_df.loc[idx, f'avg_{feat}'] = avg_prop
    
    return tmp_df

def get_elem_props(elem, vasp_valence=False):
    '''
        Extract elemental properties from pymatgen.core.periodic_table
        
        Input: element symbol, str 
    '''
    elem_props = {}
    elem_props['Z'] = periodic_table.Element(elem).Z
    elem_props['X'] = periodic_table.Element(elem).X
    elem_props['row'] = periodic_table.Element(elem).row
    elem_props['group'] = periodic_table.Element(elem).group
    elem_props['atomic_mass'] = periodic_table.Element(elem).atomic_mass
    elem_props['atomic_radius'] = periodic_table.Element(elem).atomic_radius
    elem_props['molar_volume'] = periodic_table.Element(elem).molar_volume
    elem_props['average_ionic_radius'] = periodic_table.Element(elem).average_ionic_radius
    elem_props['max_oxidation_state'] = periodic_table.Element(elem).max_oxidation_state
    elem_props['min_oxidation_state'] = periodic_table.Element(elem).min_oxidation_state
    if vasp_valence:
        elem_props['default_pp'] = df_VASP_valence[df_VASP_valence['Element'] == elem]['valency'].values[0]
        elem_props['enmax'] = df_VASP_valence[df_VASP_valence['Element'] == elem]['default_cutoff_ENMAX_(eV)'].values[0]
    return elem_props

def get_avg_prop(concentrations, prop):
    '''
        Extract features using elemental properties
        
        Input: concentrations = dict obj, prop - str (e.g. 'Z', 'X', ...)
    '''
    avg_prop = 0
    try:
        for symbol, concentraion in concentrations.items():
            avg_prop += get_elem_props(symbol, vasp_valence=True)[prop]*concentraion
        return avg_prop
    except Exception as e:
        print(f'while {symbol} is processed {e} is occured. NaN value will be returned!')
        return np.nan

def get_avg_dft_prop(concentrations, prop):
    '''
        Extract features using information about stable forms of elements
        from materials project database
        
        Input: concentrations = dict obj, prop - str (e.g. 'Z', 'X', ...)
    '''
    avg_prop = 0
    try:
        for symbol, concentraion in concentrations.items():
            ffilter = df_elements['element'] == symbol
            prop_value = df_elements[ffilter][prop].values[0]
            avg_prop += prop_value*concentraion
        return avg_prop
    except Exception as e:
        print(f'while {symbol} is processed {e} is occured. NaN value will be returned!')
        return np.nan

def get_space_groups(concentrations):
    '''
        Extract features on spacegroups 
        using information about stable forms of elements
        from materials project database
        
        Input: concentrations = dict obj
    '''
    sgd = []
    for symbol, concentraion in concentrations.items():
        ffilter = (df_elements['element'] == symbol)
        sgd.append(df_elements[ffilter]['spacegroup.symbol'].values[0])
    return sgd

def set_concentraions(df):
    '''
        Setting concentrations to extracted multi-labeled columns
        for each element presented in dataset [df]
        
        Input: any dataframe, contains column:
            - ucf: unit cell formula in pymatgen format as a standard python dictionary 
    '''
    tmp_df = df.copy()
    for idx, row in tqdm.tqdm(tmp_df.iterrows(), total=df.shape[0]):
        concentrations = row['ucf']
        for i in concentrations:
            tmp_df.loc[idx, i] = concentrations[i]
    return tmp_df

def convert_to_feats(df):
    '''
        Dataset [df] aggregation to extract all features except "hull properties"
        
        Input: any dataframe, contains columns:
            - ucf: unit cell formula in pymatgen format as a standard python dictionary 
            - dopants: python list of dopants (list of str)
            - base: str
    '''
    
    # Add average properties
    for prop in list(get_elem_props('Ti', vasp_valence=True).keys()):
        df[prop] = df['ucf'].apply(lambda x: get_avg_prop(x, prop))
    
    for prop in ['density', 'volume_per_atom', 'energy_per_atom', 'total_magnetization']:
        df[f'dft_avg_{prop}'] = df['ucf'].apply(lambda x: get_avg_dft_prop(x, prop))
    
    df['space_groups'] = df['ucf'].apply(get_space_groups)
    
    space_groups = df['space_groups']

    mlb = MultiLabelBinarizer()
    space_groups_labels = pd.DataFrame(mlb.fit_transform(space_groups),
                                       columns=mlb.classes_,
                                       index=space_groups.index)

    df = pd.concat([df, space_groups_labels], axis=1)
    df = df.drop(['space_groups'], axis='columns')
    
    # Elements concentrations
    mlb = MultiLabelBinarizer()
    elements = df['ucf'].apply(lambda x: list(x.keys()))
    elements_labels = pd.DataFrame(mlb.fit_transform(elements),
                       columns=mlb.classes_,
                       index=elements.index)

    df = pd.concat([df, elements_labels], axis=1)
    df = set_concentraions(df)
    return df

def interpolation_k_fold_cv(model, df, n_splits, prop_to_pred):
    '''
        Function to apply k-fold splitting based on alloys systems 
        from [df]. Interpolation means that each system specified in [df]
        will be splitted into different combined datasets, i.e. 
        some concentrational points will be moved to test set
        from each system at each step of validation.
        
        df: any pandas dataframe, contains columns:
            - ucf: unit cell formula in pymatgen format as a standard python dictionary 
            - dopants: python list of dopants (list of str)
            - base: str
            
        model: sklearn pipeline
        n_splits: int
        prop_to_pred: any of 'C_prime', 'B', 'E', 'G', 'c11', 'c12', 'c44'
    '''
    print('Model:', model)
    print('n_splits:', n_splits)

    test_index_b, test_index_t = [], []
    
    # Binaries
    binaries = df['dopants'].apply(lambda x: [True if len(x)==1 else False for i in x][0])

    tmp_df_g = df[binaries].copy()
    tmp_df_g['dopant'] = tmp_df_g['dopants'].apply(lambda x: x[0])
    tmp_df_g = tmp_df_g.drop(['dopants'], axis='columns')
    tmp_df_g = tmp_df_g.groupby(['base', 'dopant'])
    
    for group in tmp_df_g.groups:
        data = tmp_df_g.get_group(group)
        test_index_b.append(np.array_split(data.index.to_list(), n_splits))

    test_index_b = np.array(test_index_b).T
    test_index_b = np.array([np.concatenate(i) for i in test_index_b])
    
    # Ternaries
    ternaries = df['dopants'].apply(lambda x: [True if len(x)==2 else False for i in x][0])
    
    tmp_df_g = df[ternaries].copy()
    tmp_df_g['dopant_1'] = tmp_df_g['dopants'].apply(lambda x: x[0])
    tmp_df_g['dopant_2'] = tmp_df_g['dopants'].apply(lambda x: x[1])
    tmp_df_g = tmp_df_g.drop(['dopants'], axis='columns')
    tmp_df_g = tmp_df_g.groupby(['base', 'dopant_1', 'dopant_2'])
    
    for group in tmp_df_g.groups:
        data = tmp_df_g.get_group(group)
        test_index_t.append(np.array_split(data.index.to_list(), n_splits))
    
    test_index_t = np.array(test_index_t).T
    test_index_t = np.array([np.concatenate(i) for i in test_index_t])
    
    test_index = np.array(list(zip(test_index_b, test_index_t)))
    test_index = np.array([np.concatenate(i) for i in test_index])
    
    all_true, all_pred = [], []
    results = pd.DataFrame(columns=['r2','MAE'])
    for test_subset_index in tqdm.tqdm(test_index):
        
        test_set = df.loc[test_subset_index]
        train_set = df.loc[~df.index.isin(test_subset_index)]
        
        X_train = train_set.drop([prop_to_pred, 'base', 'dopants'], axis='columns')
        y_train = train_set[prop_to_pred]
        
        X_test = test_set.drop([prop_to_pred, 'base', 'dopants'], axis='columns')
        y_test = test_set[prop_to_pred].values
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        all_true.append(y_test)
        all_pred.append(y_pred)
        
        r2 = round(r2_score(y_test, y_pred),3)
        mae = round(mean_absolute_error(y_test, y_pred), 3)
        
        results = pd.concat([results, pd.DataFrame([{'r2':r2,'MAE':mae}])])
    
    all_true = np.concatenate(np.array(all_true))
    all_pred = np.concatenate(np.array(all_pred))
    
    results = results.reset_index(drop=True).describe()
    
    return results, all_true, all_pred, np.concatenate(test_index)

def leave_one_out_cv(model, X,y):
    '''
        Implementation of leave-one-out cross-validation
        
        model: sklearn pipeline
        X: any pandas.DataFrame         
        y: any pandas.Series obj
    '''
    true, pred = [], []
    
    for idx, data in tqdm.tqdm(X.iterrows(), total=X.shape[0]):
        
        train_X = X.drop(idx, axis='rows').copy()
        train_X = train_X.reset_index(drop=True)
        test_X = X.loc[[idx]]
        test_X = test_X.reset_index(drop=True)
        
        train_y = y.drop(idx, axis='rows').copy()
        train_y = train_y.reset_index(drop=True)
        test_y = y.loc[[idx]]
        test_y = test_y.reset_index(drop=True)
        
        model.fit(train_X, train_y)
        true.append(test_y.values)
        pred.append(model.predict(test_X))
    
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    
    res = pd.DataFrame([{
        'r2':round(r2_score(true, pred),3),
        'MAE':round(mean_absolute_error(true, pred), 3)
    }])
    
    return true, pred, res

def get_predictions(model1, model2,
                    prop_to_pred,
                    df_emto_converted, df_vasp_converted,
                    test_set): # any dataframe with 'ucf' column
    
    '''
        The same dataset as the test set will be returned 
        with a columns of predicted values
        
        model1, model2: sklearn pipelines
        prop_to_pred: any of 'C_prime', 'B', 'E', 'G', 'c11', 'c12', 'c44', str
        df_emto_converted, df_vasp_converted: pandas.DataFrames containing extracted features
        test_set: pandas.DataFrame obtaind in "system_fold_cv" function below
        
    '''
    print(f"Validated system: {test_set['base'].unique()[0]}-{test_set['dopant'].unique()[0]}")
    train_set = df_emto_converted.copy()
    
    # Code to drop any ucf from a dataframe converted to the feats
    for ucf in test_set['ucf'].to_list():
        idxs_to_drop = []
        for elem, concentration in ucf.items():
            if elem not in train_set.columns: continue
            idxs_to_drop.extend(train_set[train_set[elem] != 0].index.values)
        u, c = np.unique(idxs_to_drop, return_counts=True)
        dup = u[c > 1]        
        if len(dup) != 0: train_set = train_set.drop(dup, axis='rows')
        elif len(dup) == 0: train_set = train_set
    val_base = test_set['base'].unique()[0]
    val_dopant = test_set['dopant'].unique()[0]
    
    if val_base in train_set.columns and val_dopant in train_set.columns:
        print('Discrepancies in emto set:', 
              train_set[(train_set[val_base] != 0) & (train_set[val_dopant] != 0)].shape[0])
    else: print('Discrepancies in emto set:', 0)

    # Model 1
    X = train_set.drop(prop_to_pred, axis='columns').copy()
    y = train_set[prop_to_pred]
    model1_feats = list(X.columns)
    model1.fit(X,y)
    
    # Model 2
    predicted_EMTO = model1.predict(X)
    train_set = df_vasp_converted.copy()
    
    # Code to drop any ucf from a dataframe converted to the feats
    for ucf in test_set['ucf'].to_list():
        idxs_to_drop = []
        for elem, concentration in ucf.items():
            if elem not in train_set.columns: continue
            idxs_to_drop.extend(train_set[train_set[elem] != 0].index.values)
        u, c = np.unique(idxs_to_drop, return_counts=True)
        dup = u[c > 1]        
        if len(dup) != 0: train_set = train_set.drop(dup, axis='rows')
        elif len(dup) == 0: train_set = train_set
    
    if val_base in train_set.columns and val_dopant in train_set.columns:
        print('Discrepancies in vasp set:', 
              train_set[(train_set[val_base] != 0) & (train_set[val_dopant] != 0)].shape[0])
    else: print('Discrepancies in vasp set:', 0)
    
    X = train_set.drop(prop_to_pred, axis='columns').reset_index(drop=True)
    y = train_set[prop_to_pred].reset_index(drop=True)
    
    # Get predictions of first model for train set of second one
    X_ = X.copy()
    X_ = fill_na_feats(model1_feats, X)
    X['predicted_EMTO'] = model1.predict(X_)
    model2_feats = list(X.columns)
    model2.fit(X, y)
    
    # Get predicted data for test set
    X = mlb_feats_from_elemental_hull(test_set[['ucf']])
    X = convert_to_feats(X)
    
    X.drop(['ucf'], axis='columns', inplace=True)
    X_ = fill_na_feats(model1_feats, X)
    X = fill_na_feats(model2_feats, X) 
    predicted_EMTO = model1.predict(X_)
    X['predicted_EMTO'] = predicted_EMTO
    predicted_data = model2.predict(X)
    
    out_df = test_set.copy()
    out_df['Pred_EMTO'] = predicted_EMTO
    out_df['Pred_VASP'] = predicted_data
    
    return out_df

def system_fold_cv(tmp_df,
                   model1, model2,
                   prop_to_pred,
                   df_emto_converted, df_vasp_converted):
    
    '''
        Function to apply k-fold splitting based on alloys systems 
        from [df]. system_fold_cv means that each system specified in [df]
        will be excluded from training set at each step of validation.
        
        tmp_df: pandas.DataFrame with systems to be tested, 
                obtained from "get_set_for_system_fold_cv" function below
            
        model1, model2: sklearn pipelines
        prop_to_pred: any of 'C_prime', 'B', 'E', 'G', 'c11', 'c12', 'c44', str
        df_emto_converted, df_vasp_converted: pandas.DataFrames containing extracted features
        test_set: pandas.DataFrame obtaind in "system_fold_cv" function below
    '''
    
    all_res = pd.DataFrame()
    tmp_df_grouped_lev_1 = tmp_df.groupby(['base'])
    selected_groups_lev_1 = list(tmp_df_grouped_lev_1.groups)
    for count, base in enumerate(selected_groups_lev_1):
        count = count + 1
        test_set_lev_1 = tmp_df_grouped_lev_1.get_group(base)
        tmp_df_grouped_lev_2 = test_set_lev_1.groupby(['dopant'])
        selected_groups_lev_2 = list(tmp_df_grouped_lev_2.groups)
        for dopant in selected_groups_lev_2:
            if base == dopant: continue
            test_set = tmp_df_grouped_lev_2.get_group(dopant).copy()
            for conc in np.linspace(0.01, 0.5, 20):
                
                tmp_set = pd.DataFrame([{
                                        'ucf':{base:1-conc, dopant:conc},
                                        'True_VASP':np.nan,
                                        'True_EMTO':np.nan,
                                        'base':base,
                                        'dopant':dopant,
                                        'base_conc':1-conc,
                                        'dopant_conc':conc
                                    }])
                
                test_set = pd.concat([test_set, tmp_set])
                
            test_set = test_set.reset_index(drop=True)
            
            results = get_predictions(model1, model2,
                                        prop_to_pred,
                                        df_emto_converted, df_vasp_converted,
                                        test_set)

            all_res = pd.concat([all_res, results])
            print('-'*79)

        print(f'{count}/{len(selected_groups_lev_1)}')
        
    true_pred_emto = all_res[['True_EMTO','Pred_EMTO']]
    true_pred_emto = true_pred_emto.dropna()
    true_pred_vasp = all_res[['True_VASP','Pred_VASP']]
    true_pred_vasp = true_pred_vasp.dropna()

    metrics = pd.DataFrame([{
        'r2_EMTO':round(r2_score(true_pred_emto['True_EMTO'].values,
                                 true_pred_emto['Pred_EMTO'].values),3),
        'MAE_EMTO':round(mean_absolute_error(true_pred_emto['True_EMTO'].values,
                                             true_pred_emto['Pred_EMTO'].values),3),
        'r2_VASP':round(r2_score(true_pred_vasp['True_VASP'].values,
                                 true_pred_vasp['Pred_VASP'].values),3),
        'MAE_VASP':round(mean_absolute_error(true_pred_vasp['True_VASP'].values,
                                             true_pred_vasp['Pred_VASP'].values),3)}])

    return all_res, metrics

def get_set_for_system_fold_cv(df_emto, df_vasp, prop_to_pred):
    '''
        simple aggregation of datasets to extract common systems
    '''
    tmp_df_emto = df_emto[df_emto['dopants'].apply(lambda x: len(x)) == 1].copy()
    tmp_df_emto = tmp_df_emto[['ucf', 'base', 'dopants', prop_to_pred]]
    tmp_df_emto.columns = [f'True_EMTO' if i == prop_to_pred else i for i in tmp_df_emto.columns]
    tmp_df_emto['ucf_sorted_joined'] = tmp_df_emto['ucf'].apply(get_ucf_joined)
    tmp_df_emto['dopant'] = tmp_df_emto['dopants'].apply(lambda x: x[0])
    tmp_df_emto = tmp_df_emto.drop(['dopants'],axis='columns')

    tmp_df_vasp = df_vasp[['ucf', prop_to_pred, 'base', 'dopants']].copy()
    tmp_df_vasp.columns = [f'True_VASP' if i == prop_to_pred else i for i in tmp_df_vasp.columns]
    tmp_df_vasp['ucf_sorted_joined'] = tmp_df_vasp['ucf'].apply(get_ucf_joined)
    tmp_df_vasp['dopant'] = tmp_df_vasp['dopants'].apply(lambda x: x[0])
    tmp_df_vasp = tmp_df_vasp.drop(['dopants'],axis='columns')

    tmp_df = pd.merge(tmp_df_vasp, tmp_df_emto, on='ucf_sorted_joined', how='outer')
    tmp_df.loc[tmp_df[tmp_df['ucf_y'].isna()].index, 'ucf_y'] = tmp_df[tmp_df['ucf_y'].isna()]['ucf_x']
    tmp_df.loc[tmp_df[tmp_df['base_y'].isna()].index, 'base_y'] = tmp_df[tmp_df['base_y'].isna()]['base_x']
    tmp_df.loc[tmp_df[tmp_df['dopant_y'].isna()].index, 'dopant_y'] = tmp_df[tmp_df['dopant_y'].isna()]['dopant_x']
    tmp_df['ucf'] = tmp_df['ucf_y']
    tmp_df['base'] = tmp_df['base_y']
    tmp_df['dopant'] = tmp_df['dopant_y']
    tmp_df = tmp_df[['ucf','base','dopant','True_VASP', 'True_EMTO']]

    # Define concantrations
    for idx, data in tmp_df.iterrows():
        ucf = data['ucf']
        base = data['base']
        dopant = data['dopant']
        tmp_df.loc[idx, 'base_conc'] = ucf[base]
        tmp_df.loc[idx, 'dopant_conc'] = ucf[dopant]
    return tmp_df

def fill_na_feats(model1_feats, X):
    '''       
        Adding all available features.
        VASP set is not so wide as EMTO's one. 
        Therefore some features in comparison with EMTO set can be dissapeared.
        e.g. some groups of symmetry for some elements or presense of some elements.
    '''
    X_ = pd.DataFrame()
    for i in model1_feats:
        c = 0
        for j in X.columns:
            if i == j:
                X_[i] = X[i]
                c += 1
        if c == 0:
            X_[i] = np.nan

    X_ = X_.fillna(0)
    return X_

def get_ucf_joined(ucf):
    joined = []
    keys = sorted(list(ucf.keys()))
    for i in keys:
        joined.append(i)
        joined.append(str(round(ucf[i],2)))
    return ''.join(joined) 