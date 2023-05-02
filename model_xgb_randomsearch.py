import sys
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def rmse(a,b):
    return np.sqrt( np.mean ((a-b)**2) )
def mae(a,b):
    return np.mean( abs( a-b ) )

df_pore = pd.read_csv('zeo/PoreSize_Uff298K_coremof2019.csv')

df_info = pd.read_csv('zeo/Zeoinfo_descriptors_coremof2019.csv')
df_info.replace(np.inf,10, inplace=True)

df_vol_12 = pd.read_csv('zeo/VolumeOccupiable_Uff298K1.2_coremof2019.csv')
df_vol_18 = pd.read_csv('zeo/VolumeOccupiable_Uff298K1.8_coremof2019.csv')
df_vol_20 = pd.read_csv('zeo/VolumeOccupiable_Uff298K2.0_coremof2019.csv').rename(columns={"PO_VF":"PO_VF_2.0","POA_VF":"POA_VF_2.0","PONA_VF":"PONA_VF_2.0"})
df_vol = pd.merge(df_vol_12,df_vol_18, on="Structures",  how="left", suffixes=("_1.2", "_1.8"))
df_vol = pd.merge(df_vol,df_vol_20, on="Structures",  how="left")

df_sa_12 = pd.read_csv('zeo/SurfaceArea_Uff298K1.2_coremof2019.csv')
df_sa_18 = pd.read_csv('zeo/SurfaceArea_Uff298K1.8_coremof2019.csv')
df_sa_20 = pd.read_csv('zeo/SurfaceArea_Uff298K2.0_coremof2019.csv').rename(columns={"ASA_m2/cm3":"ASA_m2/cm3_2.0","NASA_m2/cm3":"NASA_m2/cm3_2.0", "SA_m2/cm3":"SA_m2/cm3_2.0"})
df_sa = pd.merge(df_sa_12,df_sa_18, how="left", on="Structures", suffixes=("_1.2", "_1.8"))
df_sa = pd.merge(df_sa,df_sa_20, on="Structures",  how="left")

df_chan = pd.read_csv('zeo/Channel_Uff298K1.8_coremof2019.csv')

# %%
T = 298.0
R = 8.31446261815324e-3
P_0 = 101300          # Pa
mmol2mol = 1e-3

df_xe = pd.read_csv('Screening_CoReMOF_Dataset.csv')
# df_kr = pd.read_csv('krypton_900K.csv')[['Structure_name','E_surface_B','E_surface_B_900']].rename(columns={'Structures':'Structure_name'})

# df_surf_xe = pd.read_csv('cpp_output_final_2k.csv')
# df_surf_xe['Structure_name'] = df_surf_xe['Structure_name'].str.rsplit('_',1).str[0]
# df_surf_kr = pd.read_csv('cpp_output_final_2k_krypton.csv')
# df_surf_kr['Structure_name'] = df_surf_kr['Structure_name'].str.rsplit('_',1).str[0]

df_grid_xe = pd.read_csv('output_grid_0.12_298K_100_Xe.csv')
df_grid_xe['Structures'] = df_grid_xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_xe.drop(columns=["Structure_name"],inplace=True)
df_grid_kr = pd.read_csv('output_grid_0.12_298K_100_Kr.csv')
df_grid_kr['Structures'] = df_grid_kr['Structure_name'].str.rsplit('_',n=1).str[0]
df_grid_kr.drop(columns=["Structure_name"],inplace=True)
df_psd = pd.read_csv("coremof_poredist_uff298K.csv")

df_900_Xe = pd.read_csv('output_grid_0.12_900K_Xe.csv')
df_900_Xe['Structures'] = df_900_Xe['Structure_name'].str.rsplit('_',n=1).str[0]
df_900_Xe['KH_900K'] = 1e3*R*T*df_900_Xe['Henry_coeff_molkgPa']
df_900_Xe['H_900K'] = df_900_Xe['Enthalpy_grid_kjmol']
df_900_Kr = pd.read_csv('output_grid_0.12_900K_Kr.csv')
df_900_Kr['Structures'] = df_900_Kr['Structure_name'].str.rsplit('_',n=1).str[0]
df_900_Kr['KH_900K'] = 1e3*R*T*df_900_Kr['Henry_coeff_molkgPa']
df_900_Kr['H_900K'] = df_900_Kr['Enthalpy_grid_kjmol']
df_900 = pd.merge(df_900_Xe[['Structures','KH_900K','H_900K']],df_900_Kr[['Structures','KH_900K','H_900K']], on='Structures', how='left', suffixes=("_xenon","_krypton"))

df = pd.merge(df_grid_xe,df_grid_kr, on='Structures', how='left', suffixes=("_xenon","_krypton"))

df = pd.merge(df,df_xe, on='Structures', how='left')

df = pd.merge(df,df_psd, on='Structures', how='left')
df = pd.merge(df,df_info, on='Structures', how='left')
df = pd.merge(df,df_pore, on='Structures', how='left')
df = pd.merge(df,df_chan, on='Structures', how='left')
df = pd.merge(df,df_sa, on='Structures', how='left')
df = pd.merge(df,df_vol, on='Structures', how='left')
df = pd.merge(df,df_900, on='Structures', how='left')

# %%
df['K_Xe'] = df['K_Xe_widom']
df['K_Kr'] = df['K_Kr_widom']

df['H_Xe_0'] = df['H_Xe_0_widom']
df['H_Kr_0'] = df['H_Kr_0_widom']

df['Delta_H_0'] = df['H_Xe_0'] - df['H_Kr_0']
# Need for enthalpy of krypton maybe ?

df['s_2080_log'] = np.log10(df['s_2080'])

df['Delta_H_2080'] = df['H_Xe_2080'] - df['H_Kr_2080']

df['s_0'] = df['Henry_coeff_molkgPa_xenon']/df['Henry_coeff_molkgPa_krypton']
df['s_0'] = df['s_0'].replace(0,np.nan)

df['s_2080'] = df['s_2080'].replace(0,np.nan)

df.replace([np.inf,-np.inf],np.nan,inplace=True)
df = df[~(df['DISORDER']=='DISORDER')]
print(df.shape[0])

# %%
#### RESTRICTION on non-radioactive ASR 3D-MOFs 
df_data = df[(df['framework_mean_dim']>2)] #3D
df_data = df_data[(df_data['solvent_removed']==1)] #ASR
df_data = df_data[(df_data['C%']>0)&(df_data['metal%']>0)] #MOF
df_data = df_data[(df_data['radioactive%']==0)] #nonradioactive
print(df_data.shape[0])

#### RESTRICTION on materials porous enough for xenon
df_data = df_data[df_data['D_i_vdw_uff298']>4]
print(df_data.shape[0])

# %%
df_data.rename(columns={'Framework Mass [g/mol]':"mass_g_mol", 'Framework Density [kg/m^3]':'density_kg_m3'}, inplace=True)

df_data['G_2080'] = -R*T*np.log(df_data['s_2080'])
df_data['G_0'] = -R*T*np.log(df_data['s_0'])
df_data['G_0_widom'] = -R*T*np.log(df_data['s_0_widom'])
df_data['pore_dist_modality'] = (df_data['pore_dist_skewness']**2+1)/df_data['pore_dist_kurtosis']

df_data['G_900K'] = -R*900*np.log(df_data['KH_900K_xenon']/df_data['KH_900K_krypton'])

df_data['G_Xe_900K'] = -R*900*np.log(df_data['density_kg_m3'] * df_data['KH_900K_xenon'])
df_data['G_Kr_900K'] = -R*900*np.log(df_data['density_kg_m3'] * df_data['KH_900K_krypton'])

df_data["delta_G0_298_900"] = df_data['G_900K']*298/900 - df_data['G_0']
df_data["delta_H0_Xe_298_900"] = df_data['Enthalpy_grid_kjmol_xenon'] - df_data['H_900K_xenon'] 
df_data['delta_TS0_298_900'] = df_data['delta_G0_298_900'] - df_data['delta_H0_Xe_298_900']

df_data['N/O'] = (df_data['N%']/df_data['O%']).replace(np.inf,10)
df_data['DU%'] = 100/df_data['atoms_count'] + df_data["C%"] - 0.5*(df_data['H%'] + df_data['halogen%'] - df_data['N%'])
df_data['DU_C'] = df_data['DU%']/df_data['C%']
df_data['DU'] = 1 + df_data['atoms_count'] * ( df_data["C%"] - 0.5*(df_data['H%'] + df_data['halogen%'] - df_data['N%']) )/100

df_data['LCD/PLD'] = df_data['D_i_vdw_uff298']/df_data['D_f_vdw_uff298']

df_data['SA_1.2'] = df_data['ASA_1.2'] + df_data['NASA_1.2']
df_data['delta_SA'] = df_data['SA_m2/cm3_1.8'] - df_data['SA_m2/cm3_2.0']
df_data['delta_VF_12_20'] = df_data['PO_VF_1.2'] - df_data['PO_VF_2.0']
df_data['delta_VF_18_20'] = df_data['PO_VF_1.8'] - df_data['PO_VF_2.0']

df_data['delta_pore'] = df_data['D_i_vdw_uff298'] - df_data['pore_dist_mean']

df_data['enthalpy_modality'] = (df_data['enthalpy_skew']**2+1)/df_data['enthalpy_kurt']

df_data['pore_dist_neff_log10'] = np.log10(df_data['pore_dist_neff'])

X_columns = [
'G_0', 
'G_Xe_900K',
'G_Kr_900K', 
'G_900K', 
"delta_G0_298_900",
"delta_H0_Xe_298_900",
"delta_TS0_298_900",
"enthalpy_std_xenon",
"enthalpy_std_krypton",
"enthalpy_skew",
# "enthalpy_kurt",
"enthalpy_modality",
"mean_grid_xenon",
"mean_grid_krypton",
"std_grid_xenon",
"std_grid_krypton",
'ASA_m2/cm3_1.2',
"delta_VF_18_20",
'PO_VF_2.0',
'D_i_vdw_uff298', 
'delta_pore',
'D_f_vdw_uff298', 
'pore_dist_mean', # extremely corr to LCD
'pore_dist_std',
'pore_dist_skewness', 
'pore_dist_kurtosis',
# 'pore_dist_neff_log10',
"pore_dist_neff",
'pore_dist_modality',
# 'mass_g_mol', 
# 'Density_1.2',
# 'atoms_count', 
'C%',
# 'H%',
'O%',
'N%',
# 'chan_mean_dim',
# 'chan_count', # divide by symmetry 
# 'DU',
# "delta_VF_12_20",
# 'SA_m2/cm3_1.2',
# 'SA_m2/cm3_1.8', 
# 'SA_m2/cm3_2.0', 
# 'ASA_m2/cm3_1.8', 
# 'ASA_m2/cm3_2.0', 
# 'POA_VF_1.2','POA_VF_1.8','POA_VF_2.0',
# 'PO_VF_1.2',
# 'PO_VF_1.8',
# 'LCD/PLD',
# 'DU_C','DU%',
# 'halogen%','metalloid%','ametal%',
# 'metal%',
# 'N/O',
# 'M/C',
# 'M/O'
]

y_column = ['G_2080']

# Remove nan values (search for other solutions)
# Maybe more suitable to replace by fillna with median or null values
print(len(df_data))
df_data.dropna(subset=X_columns,inplace=True)
print(len(df_data))
df_data.dropna(subset=y_column,inplace=True)
print(len(df_data))

X, y = df_data[X_columns], df_data[y_column]

random_state=123
test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

nfold = int(1/test_size)
seed=random_state

xgbr = xgb.XGBRegressor(seed = seed)

params_search = { 
           'n_estimators': [1500], 
           'max_depth': [5,6],
           'learning_rate': [0.02,0.04,0.06,0.08],
           'colsample_bytree': np.arange(0.6, 1.0, 0.05),
        #    'colsample_bytree': [0.8],
           'colsample_bylevel': np.arange(0.6, 1.0, 0.05),
        #    'colsample_bylevel': [0.6],
           'alpha': np.arange(0, 4, 0.2),
           'lambda': [0,0.5,1],
           'subsample': np.arange(0.6, 0.95, 0.05),
        #    'subsample': [0.6,0.7,0.8],
         }

s = 1
for key in params_search.keys():
    s *= len(params_search[key])
print(s)

kf = KFold(n_splits=5, shuffle = True, random_state = 1001)

clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params_search,
                         scoring='neg_mean_squared_error',
                         n_iter=30000,
                         verbose=1,
                         cv=kf.split(X_train,y_train.values), 
                         n_jobs=-1,
                         random_state = 1001)

clf.fit(X_train, y_train.values)

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", np.sqrt(-clf.best_score_))

original_stdout = sys.stdout # Save a reference to the original standard output

with open('filename.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", np.sqrt(-clf.best_score_))
    sys.stdout = original_stdout # Reset the standard output to its original value
