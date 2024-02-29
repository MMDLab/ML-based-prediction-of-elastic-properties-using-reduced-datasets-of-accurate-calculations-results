# Common libs
import pandas as pd
from sklearn import metrics
import numpy as np
import torch, pickle

# Pymatgen & matminer
from pymatgen.core import Composition
from pymatgen.core import periodic_table
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import alloy

class MultiModel:
    def __init__(self, model1, model2):
        super(MultiModel, self).__init__()
        self.model_emto = model1
        self.model_vasp = model2
        self.pred_names = 'Model was not trained'
        self.training_metrics = 'Model was not trained'
        self.emto_feats = 'Model was not trained'
        self.vasp_feats = 'Model was not trained'
        self.version = '20240219'
        self.custom_emto_feats = 'Model was not trained'
        self.custom_vasp_feats = 'Model was not trained'

    def __repr__(self):
        r = [
            f'VERSION: {self.version}',
            f'Model 1: {str(self.model_emto)}',
            f'Model 2: {str(self.model_vasp)}',
            f'To predict: {self.pred_names}',
            f'EMTO feats: {self.emto_feats}',
            f'VASP feats: {self.vasp_feats}',
        ]
        return '\n'.join(r)

    @staticmethod
    def get_X_y(data=None, method=None, only_X=False):
        
        unused_cols = ['ucf','sorted_formula', 'base','composition', 'Weight Fraction','Atomic Fraction']
        unused_cols.extend([i for i in data.columns if ('EMTO' in i)])
        unused_cols.extend([i for i in data.columns if ('VASP' in i)])
        
        uc = []
        for col in data.columns:
            if col not in unused_cols: uc.append(col)
        
        if not only_X:
            predicted_cols = []
            for pred_prop in ['B', 'E', 'G', 'C_prime', 'c11', 'c12', 'c44']:
                if f'{pred_prop}_{method}' in data.columns:
                    predicted_cols.append(f'{pred_prop}_{method}')
            if len(predicted_cols) == 0:
                raise Exception('No predicted properties in the dataset')
                
            idxs = data[predicted_cols].dropna().index
            X = data.loc[idxs, uc].copy()
            y = data.loc[idxs, predicted_cols].copy()
            return X, y
        else: return data[uc]

    def set_feats(self, emto_feats=None, vasp_feats=None):
        if emto_feats != None:
            self.custom_emto_feats = emto_feats
        if vasp_feats != None:
            self.custom_vasp_feats = vasp_feats
    
    def fit(self, data):
        # Train data
        X_emto, y_emto = self.get_X_y(data=data, method='EMTO', only_X=False)
        self.emto_feats = X_emto.columns
        if self.custom_emto_feats != 'Model was not trained':
            X_emto = X_emto[self.custom_emto_feats]
        
        X_vasp, y_vasp = self.get_X_y(data=data, method='VASP', only_X=False)
        self.vasp_feats = X_vasp.columns
        if self.custom_vasp_feats != 'Model was not trained':
            X_vasp = X_vasp[self.custom_vasp_feats]

        # Columns to predict
        self.pred_names = {}
        self.pred_names['EMTO'] = y_emto.columns.tolist()
        self.pred_names['VASP'] = y_vasp.columns.tolist()

        # Fit emto part
        self.model_emto.fit(X_emto, y_emto)
        self.emto_feats = X_emto.columns

        # Fit vasp part
        X_emto = data.loc[X_vasp.index,self.emto_feats]
        add_cols = [f"pred_feat_{i.replace('_EMTO','')}" for i in y_emto.columns]
        emto_pred = self.model_emto.predict(X_emto)
        if len(emto_pred.shape) == 1:
            emto_pred = emto_pred.reshape(-1,1)
        X_vasp[add_cols] =  emto_pred
        
        self.model_vasp.fit(X_vasp, y_vasp)
        self.vasp_feats = X_vasp.columns

    def predict(self, X:pd.DataFrame=None):
        # EMTO feats
        X = self.get_X_y(data=X, method=None, only_X=True)
        X_emto = X[self.emto_feats]

        # VASP feats
        X_vasp = X.copy()
        add_cols = [f"pred_feat_{i.replace('_EMTO','')}" for i in self.pred_names['EMTO']]
        X_vasp[add_cols] =  self.model_emto.predict(X_emto)
        X_vasp = X_vasp[self.vasp_feats]

        # Collect results
        res = pd.DataFrame()
        res['orig_index'] = X.index
        res[[i.replace('feat', 'EMTO') for i in add_cols]] = X_vasp[add_cols].values
        res[[i.replace('feat', 'VASP') for i in add_cols]] = self.model_vasp.predict(X_vasp)
        
        return res

    @staticmethod
    def get_metrics(true_data=None, pred_data=None):

        # Avoid r2 error when only 1 sample is presented
        if true_data.shape[0] > 1:
            r2 = metrics.r2_score(true_data,pred_data)
        else:
            r2 = np.nan
        
        scores = {
            'r2':r2,
            'mae': metrics.mean_absolute_error(true_data,pred_data),
            'max_abs_err':metrics.max_error(true_data,pred_data),
        }
        return scores

    def get_feature_importance(self):
        imps = pd.DataFrame()
        imps['feat'] = self.emto_feats
        imps['EMTO'] = self.model_emto.get_feature_importance()
        imps = imps.set_index(['feat'])
        for f,i in zip(self.vasp_feats, self.model_vasp.get_feature_importance()):
            imps.loc[f, 'VASP'] = i
        return imps

    def get_elem_props(self, elem, vasp_valence):
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
        
        df_VASP_valence = pd.read_csv('./data/Default_PAW_potentials_VASP.csv', sep=';')
        
        if vasp_valence:
            elem_props['default_pp_valency'] = df_VASP_valence[df_VASP_valence['Element'] == elem]['valency'].values[0]
            elem_props['enmax'] = df_VASP_valence[df_VASP_valence['Element'] == elem]['default_cutoff_ENMAX_(eV)'].values[0]
        return elem_props
    
    def get_avg_prop(self, concentrations, prop):
        ''' concentrations = dict obj '''
        avg_prop = 0
        try:
            for symbol, concentraion in concentrations.items():
                avg_prop += self.get_elem_props(symbol, vasp_valence=True)[prop]*concentraion
            return avg_prop
        except Exception as e:
            print(f'while {symbol} is processed {e} is occured. NaN value will be returned!')
            return np.nan

    def get_matminer_feats(self, df):
        tmp = df.copy()
        tmp = StrToComposition().featurize_dataframe(tmp, "sorted_formula")
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        tmp = ep_feat.featurize_dataframe(tmp, col_id="composition", ignore_errors=True)
        WA_feat = alloy.WenAlloys()
        tmp = WA_feat.featurize_dataframe(tmp, col_id="composition", ignore_errors=True)
        YSS_feat = alloy.Miedema()
        tmp = YSS_feat.featurize_dataframe(tmp, col_id="composition", ignore_errors=True)
        return tmp

    def get_avg_hull(self, concentrations=None, column=None, hull_feats=None):
        avg_prop = 0
        try:
            for symbol, concentraion in concentrations.items():
                avg_prop += hull_feats.loc[symbol,column]*concentraion
            return avg_prop
        except Exception as e:
            print(f'while {symbol} is processed {e} is occured. NaN value will be returned!')
            return np.nan

    def featurize_df(self, df):

        if 'sorted_formula' not in df.columns:
            raise ValueError('missed column sorted_formula')
        
        df_feats = df.copy()
        f2c = lambda x: Composition(x).as_dict()
        df_feats['ucf'] = df_feats['sorted_formula'].apply(f2c)
        
        # Feats
        ## Common feats
        props = list(self.get_elem_props(elem='Ti', vasp_valence=True).keys())
        for p in props:
            df_feats[f'feat_periodic_{p}'] = df_feats['ucf'].apply(lambda x: self.get_avg_prop(concentrations=x, prop=p))

        ## Matminer feats
        df_feats = self.get_matminer_feats(df=df_feats)
    
        ## Hull feats
        hull_feats = pd.read_json('./data/hull_feats.json')
        feats_cols = hull_feats.columns
        for p in feats_cols:
            df_feats[f'feat_hull_{p}'] = df_feats['ucf'].apply(lambda x: self.get_avg_hull(concentrations=x, column=p, hull_feats=hull_feats))

        return df_feats

    def predict_from_file(self, file=None):
        with open(file, 'r') as f:
            fc = f.readlines()
        fc = [i.strip() for i in fc]
        df = pd.DataFrame(fc, columns=['sorted_formula'])
        data = self.featurize_df(df=df)
        preds = self.predict(data) 
        preds['formula'] = data.loc[preds['orig_index'].values]['sorted_formula'].values
        cols = ['formula']
        for col in preds.columns:
            if 'VASP' in col: cols.append(col)
        preds = preds[cols]
        preds.columns = [i.replace('pred_VASP_', '') for i in cols]
        return preds
        
    def save_model(self, filename='model.dump'):
        with open(filename, 'wb') as f:
            pickle.dump(self,f)
        return f'Model saved as {filename}'

    @staticmethod
    def load_model(filename:str=None):
        filehandler = open(filename, 'rb') 
        return pickle.load(filehandler)
