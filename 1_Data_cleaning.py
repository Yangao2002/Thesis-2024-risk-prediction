import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

pd.set_option('future.no_silent_downcasting', True)

#--------------------------------- Loading data ---------------------------------#

def read_data(base_folder, base_name):
    base_path = '/Users/xxx/Desktop/LISS data/' 
    file_path = f"{base_path}{base_folder}/{base_name}.dta"  
    dataset = pd.read_stata(file_path) 
    return dataset

#--------------------------------- Health ---------------------------------#
#Row data

health_23 = read_data('health', 'ch23p_EN_1.0p')  
health_22 = read_data('health', 'ch22o_EN_1.0p') 
health_21 = read_data('health', 'ch21n_EN_1.0p')
health_20 = read_data('health', 'ch20m_EN_1.0p') 
health_19 = read_data('health', 'ch19l_EN_1.0p')
health_18 = read_data('health', 'ch18k_EN_1.0p') 
health_17 = read_data('health', 'ch17j_EN_1.0p') 

#--------------- 2023 ---------------#

# Data preprocessing: Extract, convert, and clean data structure

health_23['year'] = health_23['ch23p_m'].astype(str).str[:4].astype(int)
health_23.loc[:, 'paid_job'] = health_23['ch23p003'].map({'has paid job': 1}).fillna(0)
health_23.loc[:, 'health'] = health_23['ch23p004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_23.loc[:, 'anxiety'] = health_23['ch23p011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_23.loc[:, 'down'] = health_23['ch23p012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_23.loc[:, 'calmness'] = health_23['ch23p013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_23.loc[:, 'depression'] = health_23['ch23p014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_23.loc[:, 'happiness'] = health_23['ch23p015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_23.loc[:, 'chest_pain'] = health_23['ch23p071'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 's_breath'] = health_23['ch23p072'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'coughing'] = health_23['ch23p073'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'headache'] = health_23['ch23p075'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'fatigue'] = health_23['ch23p076'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'sleeping_prb'] = health_23['ch23p077'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'angina'] = health_23['ch23p080'].map({'yes': 1, 'no': 0}).fillna(0)
health_23.loc[:, 'heart_dis'] = health_23['ch23p081'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'hypertension'] = health_23['ch23p082'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'h_cholesterol'] = health_23['ch23p083'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'stroke'] = health_23['ch23p084'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'diabetes'] = health_23['ch23p085'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'cancer'] = health_23['ch23p089'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'benign_tumor'] = health_23['ch23p096'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'hs_smk'] = health_23['ch23p125'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'curr_smk'] = health_23['ch23p126'].map({'yes': 1}).fillna(0)

health_23['ch23p133'] = health_23['ch23p133'].str.strip()
health_23.loc[:, 'drink'] = health_23['ch23p133'].fillna('none')
health_23.loc[:, 'drink'] = health_23['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_23.loc[:, 'drink_7day'] = health_23['ch23p134'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_h_chol'] = health_23['ch23p169'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_hypertension'] = health_23['ch23p170'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_MI'] = health_23['ch23p171'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_heart_dis'] = health_23['ch23p172'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_asthma'] = health_23['ch23p173'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_diabetes'] = health_23['ch23p174'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_joint_pain'] = health_23['ch23p175'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_pains'] = health_23['ch23p176'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_sleeping_prb'] = health_23['ch23p177'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'med_anxiety'] = health_23['ch23p178'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'no_med'] = health_23['ch23p184'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'hospital'] = health_23['ch23p239'].map({'yes': 1}).fillna(0)
health_23['LOS'] = health_23['ch23p230'].fillna(0)
health_23.loc[:, 'operation'] = health_23['ch23p231'].map({'yes': 1}).fillna(0)
health_23.loc[:, 'insur'] = health_23['ch23p239'].map({'yes': 1}).fillna(0)
health_23['obli_risk'] = health_23['ch23p260'].str.extract('(\d+)').fillna(0).astype(int)
health_23['ch23p263'] = health_23['ch23p263'].str.strip() # Deal with the space
health_23.loc[:, 'healthcare_allo'] = health_23['ch23p263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_23['healthcare_num'] = health_23['ch23p264'].map({
    "I don't know": 0
}).fillna(health_23['ch23p264']).astype(float)
health_23['healthcare_num'] = health_23['healthcare_num'].fillna(0).astype(float)


# Rename

health_23 = health_23.rename(columns={
    'ch23p001': 'sex',
    'ch23p002': 'age',
    'ch23p016':'height',
    'ch23p017':'weight',
    'ch23p130':'cig_num',  
    'ch23p131':'pipe_num', 
    'ch23p132':'cigar_num',  
    'ch23p266':'ecig_num', 
    'ch23p133':'drink_freq',
    'ch23p255':'start_date',
    'ch23p256':'start_time',
    'ch23p257':'end_date',
    'ch23p258':'end_time'  
})


list_health = ['nomem_encr',
           'age',
           'height',
           'weight',
           'year',
           'sex',
           'paid_job',
           'health',
           'anxiety',
           'down',
           'calmness',
           'depression',
           'happiness',
           'chest_pain',
           's_breath',
           'coughing',
           'headache',
           'fatigue',
           'sleeping_prb',
           'angina',
           'heart_dis',
           'hypertension',
           'h_cholesterol',
           'stroke',
           'diabetes',
           'cancer',
           'benign_tumor',
           'hs_smk',
           'curr_smk',
           'cig_num',
           'pipe_num',
           'cigar_num', 
           'ecig_num', 
           'drink',
           'drink_7day',
           'drink_freq',
           'med_h_chol',
           'med_hypertension',
           'med_MI',
           'med_heart_dis',
           'med_asthma',
           'med_diabetes',
           'med_joint_pain',
           'med_pains',
           'med_sleeping_prb',
           'med_anxiety',
           'no_med',
           'hospital',
           'LOS',
           'operation',
           'insur',
           'obli_risk',
           'healthcare_allo',
           'healthcare_num',
           'start_date',
           'start_time',
           'end_date',
           'end_time' 
           ]

health_23_2 = health_23[list_health]


#--------------------------------- 2022 ---------------------------------#

# Data preprocessing: Extract, convert, and clean data structure

health_22['year'] = health_22['ch22o_m'].astype(str).str[:4].astype(int)
health_22.loc[:, 'paid_job'] = health_22['ch22o003'].map({'has paid job': 1}).fillna(0)
health_22.loc[:, 'health'] = health_22['ch22o004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_22.loc[:, 'anxiety'] = health_22['ch22o011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_22.loc[:, 'down'] = health_22['ch22o012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_22.loc[:, 'calmness'] = health_22['ch22o013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_22.loc[:, 'depression'] = health_22['ch22o014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_22.loc[:, 'happiness'] = health_22['ch22o015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_22.loc[:, 'chest_pain'] = health_22['ch22o071'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 's_breath'] = health_22['ch22o072'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'coughing'] = health_22['ch22o073'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'headache'] = health_22['ch22o075'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'fatigue'] = health_22['ch22o076'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'sleeping_prb'] = health_22['ch22o077'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'angina'] = health_22['ch22o080'].map({'yes': 1, 'no': 0}).fillna(0)
health_22.loc[:, 'heart_dis'] = health_22['ch22o081'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'hypertension'] = health_22['ch22o082'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'h_cholesterol'] = health_22['ch22o083'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'stroke'] = health_22['ch22o084'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'diabetes'] = health_22['ch22o085'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'cancer'] = health_22['ch22o089'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'benign_tumor'] = health_22['ch22o096'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'hs_smk'] = health_22['ch22o125'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'curr_smk'] = health_22['ch22o126'].map({'yes': 1}).fillna(0)

health_22['ch22o133'] = health_22['ch22o133'].str.strip()
health_22.loc[:, 'drink'] = health_22['ch22o133'].fillna('none')
health_22.loc[:, 'drink'] = health_22['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_22.loc[:, 'drink_7day'] = health_22['ch22o134'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_h_chol'] = health_22['ch22o169'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_hypertension'] = health_22['ch22o170'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_MI'] = health_22['ch22o171'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_heart_dis'] = health_22['ch22o172'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_asthma'] = health_22['ch22o173'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_diabetes'] = health_22['ch22o174'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_joint_pain'] = health_22['ch22o175'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_pains'] = health_22['ch22o176'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_sleeping_prb'] = health_22['ch22o177'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'med_anxiety'] = health_22['ch22o178'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'no_med'] = health_22['ch22o184'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'hospital'] = health_22['ch22o229'].map({'yes': 1}).fillna(0)
health_22['LOS'] = health_22['ch22o230'].fillna(0)
health_22.loc[:, 'operation'] = health_22['ch22o231'].map({'yes': 1}).fillna(0)
health_22.loc[:, 'insur'] = health_22['ch22o239'].map({'yes': 1}).fillna(0)
health_22['obli_risk'] = health_22['ch22o260'].str.extract('(\d+)').fillna(0).astype(int)
health_22['ch22o263'] = health_22['ch22o263'].str.strip() # Deal with the space
health_22.loc[:, 'healthcare_allo'] = health_22['ch22o263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_22['healthcare_num'] = health_22['ch22o264'].map({"I don't know": 0}).fillna(health_22['ch22o264']).astype(float)
health_22['healthcare_num'] = health_22['healthcare_num'].fillna(0).astype(float)

health_22 = health_22.rename(columns={
    'ch22o001': 'sex',
    'ch22o002': 'age',
    'ch22o016':'height',
    'ch22o017':'weight',
    'ch22o130':'cig_num',  
    'ch22o131':'pipe_num', 
    'ch22o132':'cigar_num',  
    'ch22o266':'ecig_num', 
    'ch22o133':'drink_freq',
    'ch22o255':'start_date',
    'ch22o256':'start_time',
    'ch22o257':'end_date',
    'ch22o258':'end_time'  
})

health_22_2 = health_22[list_health]

#--------------------------------- 2021 ---------------------------------#

# Data preprocessing: Extract, convert, and clean data structure

health_21['year'] = health_21['ch21n_m'].astype(str).str[:4].astype(int)
health_21['ch21n001'] = health_21['ch21n001'].astype(str)
health_21.loc[:, 'paid_job'] = health_21['ch21n003'].map({'has paid job': 1}).fillna(0)
health_21.loc[:, 'health'] = health_21['ch21n004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_21.loc[:, 'anxiety'] = health_21['ch21n011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_21.loc[:, 'down'] = health_21['ch21n012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_21.loc[:, 'calmness'] = health_21['ch21n013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_21.loc[:, 'depression'] = health_21['ch21n014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_21.loc[:, 'happiness'] = health_21['ch21n015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_21.loc[:, 'chest_pain'] = health_21['ch21n071'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 's_breath'] = health_21['ch21n072'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'coughing'] = health_21['ch21n073'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'headache'] = health_21['ch21n075'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'fatigue'] = health_21['ch21n076'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'sleeping_prb'] = health_21['ch21n077'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'angina'] = health_21['ch21n080'].map({'yes': 1, 'no': 0}).fillna(0)
health_21.loc[:, 'heart_dis'] = health_21['ch21n081'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'hypertension'] = health_21['ch21n082'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'h_cholesterol'] = health_21['ch21n083'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'stroke'] = health_21['ch21n084'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'diabetes'] = health_21['ch21n085'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'cancer'] = health_21['ch21n089'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'benign_tumor'] = health_21['ch21n096'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'hs_smk'] = health_21['ch21n125'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'curr_smk'] = health_21['ch21n126'].map({'yes': 1}).fillna(0)

health_21['ch21n133'] = health_21['ch21n133'].str.strip()
health_21.loc[:, 'drink'] = health_21['ch21n133'].fillna('none')
health_21.loc[:, 'drink'] = health_21['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_21.loc[:, 'drink_7day'] = health_21['ch21n134'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_h_chol'] = health_21['ch21n169'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_hypertension'] = health_21['ch21n170'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_MI'] = health_21['ch21n171'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_heart_dis'] = health_21['ch21n172'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_asthma'] = health_21['ch21n173'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_diabetes'] = health_21['ch21n174'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_joint_pain'] = health_21['ch21n175'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_pains'] = health_21['ch21n176'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_sleeping_prb'] = health_21['ch21n177'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'med_anxiety'] = health_21['ch21n178'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'no_med'] = health_21['ch21n184'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'hospital'] = health_21['ch21n229'].map({'yes': 1}).fillna(0)
health_21['LOS'] = health_21['ch21n230'].fillna(0)
health_21.loc[:, 'operation'] = health_21['ch21n231'].map({'yes': 1}).fillna(0)
health_21.loc[:, 'insur'] = health_21['ch21n239'].map({'yes': 1}).fillna(0)
health_21['obli_risk'] = health_21['ch21n260'].str.extract('(\d+)').fillna(0).astype(int)
health_21['ch21n263'] = health_21['ch21n263'].str.strip() # Deal with the space
health_21.loc[:, 'healthcare_allo'] = health_21['ch21n263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_21['healthcare_num'] = health_21['ch21n264'].map({"I don't know": 0}).fillna(health_21['ch21n264']).astype(float)
health_21['healthcare_num'] = health_21['healthcare_num'].fillna(0).astype(float)

health_21 = health_21.rename(columns={
    'ch21n001': 'sex',
    'ch21n002':'age',
    'ch21n016':'height',
    'ch21n017':'weight',

    'ch21n130':'cig_num',  
    'ch21n131':'pipe_num', 
    'ch21n132':'cigar_num',
    'ch21n266':'ecig_num',
    'ch21n133':'drink_freq',
    'ch21n255':'start_date',
    'ch21n256':'start_time',
    'ch21n257':'end_date',
    'ch21n258':'end_time'  
})

health_21_2 = health_21[list_health]


#--------------------------------- 2020 ---------------------------------#

# Data preprocessing: Extract, convert, and clean data structure

health_20['year'] = health_20['ch20m_m'].astype(str).str[:4].astype(int)
health_20.loc[:, 'paid_job'] = health_20['ch20m003'].map({'has paid job': 1}).fillna(0)
health_20.loc[:, 'health'] = health_20['ch20m004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_20.loc[:, 'anxiety'] = health_20['ch20m011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_20.loc[:, 'down'] = health_20['ch20m012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_20.loc[:, 'calmness'] = health_20['ch20m013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_20.loc[:, 'depression'] = health_20['ch20m014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_20.loc[:, 'happiness'] = health_20['ch20m015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_20.loc[:, 'chest_pain'] = health_20['ch20m071'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 's_breath'] = health_20['ch20m072'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'coughing'] = health_20['ch20m073'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'headache'] = health_20['ch20m075'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'fatigue'] = health_20['ch20m076'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'sleeping_prb'] = health_20['ch20m077'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'angina'] = health_20['ch20m080'].map({'yes': 1, 'no': 0}).fillna(0)
health_20.loc[:, 'heart_dis'] = health_20['ch20m081'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'hypertension'] = health_20['ch20m082'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'h_cholesterol'] = health_20['ch20m083'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'stroke'] = health_20['ch20m084'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'diabetes'] = health_20['ch20m085'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'cancer'] = health_20['ch20m089'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'benign_tumor'] = health_20['ch20m096'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'hs_smk'] = health_20['ch20m125'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'curr_smk'] = health_20['ch20m126'].map({'yes': 1}).fillna(0)

health_20['ch20m133'] = health_20['ch20m133'].str.strip()
health_20.loc[:, 'drink'] = health_20['ch20m133'].fillna('none')
health_20.loc[:, 'drink'] = health_20['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_20.loc[:, 'drink_7day'] = health_20['ch20m134'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_h_chol'] = health_20['ch20m169'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_hypertension'] = health_20['ch20m170'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_MI'] = health_20['ch20m171'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_heart_dis'] = health_20['ch20m172'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_asthma'] = health_20['ch20m173'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_diabetes'] = health_20['ch20m174'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_joint_pain'] = health_20['ch20m175'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_pains'] = health_20['ch20m176'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_sleeping_prb'] = health_20['ch20m177'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'med_anxiety'] = health_20['ch20m178'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'no_med'] = health_20['ch20m184'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'hospital'] = health_20['ch20m229'].map({'yes': 1}).fillna(0)
health_20['LOS'] = health_20['ch20m230'].fillna(0)
health_20.loc[:, 'operation'] = health_20['ch20m231'].map({'yes': 1}).fillna(0)
health_20.loc[:, 'insur'] = health_20['ch20m239'].map({'yes': 1}).fillna(0)
health_20['obli_risk'] = health_20['ch20m260'].str.extract('(\d+)').fillna(0).astype(int)
health_20['ch20m263'] = health_20['ch20m263'].str.strip() # deal with the space
health_20.loc[:, 'healthcare_allo'] = health_20['ch20m263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_20['healthcare_num'] = health_20['ch20m264'].map({"I don't know": 0}).fillna(health_20['ch20m264']).astype(float)
health_20['healthcare_num'] = health_20['healthcare_num'].fillna(0).astype(float)

health_20 = health_20.rename(columns={
    'ch20m001': 'sex',
    'ch20m002': 'age',
    'ch20m016':'height',
    'ch20m017':'weight',

    'ch20m130':'cig_num',  
    'ch20m131':'pipe_num', 
    'ch20m132':'cigar_num', 
    'ch20m266':'ecig_num',
    'ch20m133':'drink_freq',
    'ch20m255':'start_date',
    'ch20m256':'start_time',
    'ch20m257':'end_date',
    'ch20m258':'end_time'  
})

health_20_2 = health_20[list_health]

#--------------------------------- 2019 ---------------------------------#

# Data preprocessing: Extract, convert, and clean data structure

health_19['year'] = health_19['ch19l_m'].astype(str).str[:4].astype(int)
health_19.loc[:, 'paid_job'] = health_19['ch19l003'].map({'has paid job': 1}).fillna(0)
health_19.loc[:, 'health'] = health_19['ch19l004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_19.loc[:, 'anxiety'] = health_19['ch19l011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_19.loc[:, 'down'] = health_19['ch19l012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_19.loc[:, 'calmness'] = health_19['ch19l013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_19.loc[:, 'depression'] = health_19['ch19l014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_19.loc[:, 'happiness'] = health_19['ch19l015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_19.loc[:, 'chest_pain'] = health_19['ch19l071'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 's_breath'] = health_19['ch19l072'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'coughing'] = health_19['ch19l073'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'headache'] = health_19['ch19l075'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'fatigue'] = health_19['ch19l076'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'sleeping_prb'] = health_19['ch19l077'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'angina'] = health_19['ch19l080'].map({'yes': 1, 'no': 0}).fillna(0)
health_19.loc[:, 'heart_dis'] = health_19['ch19l081'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'hypertension'] = health_19['ch19l082'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'h_cholesterol'] = health_19['ch19l083'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'stroke'] = health_19['ch19l084'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'diabetes'] = health_19['ch19l085'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'cancer'] = health_19['ch19l089'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'benign_tumor'] = health_19['ch19l096'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'hs_smk'] = health_19['ch19l125'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'curr_smk'] = health_19['ch19l126'].map({'yes': 1}).fillna(0)

health_19['ch19l133'] = health_19['ch19l133'].str.strip()
health_19.loc[:, 'drink'] = health_19['ch19l133'].fillna('none')
health_19.loc[:, 'drink'] = health_19['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_19.loc[:, 'drink_7day'] = health_19['ch19l134'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_h_chol'] = health_19['ch19l169'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_hypertension'] = health_19['ch19l170'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_MI'] = health_19['ch19l171'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_heart_dis'] = health_19['ch19l172'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_asthma'] = health_19['ch19l173'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_diabetes'] = health_19['ch19l174'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_joint_pain'] = health_19['ch19l175'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_pains'] = health_19['ch19l176'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_sleeping_prb'] = health_19['ch19l177'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'med_anxiety'] = health_19['ch19l178'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'no_med'] = health_19['ch19l184'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'hospital'] = health_19['ch19l229'].map({'yes': 1}).fillna(0)
health_19['LOS'] = health_19['ch19l230'].fillna(0)
health_19.loc[:, 'operation'] = health_19['ch19l231'].map({'yes': 1}).fillna(0)
health_19.loc[:, 'insur'] = health_19['ch19l239'].map({'yes': 1}).fillna(0)
health_19['obli_risk'] = health_19['ch19l260'].str.extract('(\d+)').fillna(0).astype(int)
health_19['ch19l263'] = health_19['ch19l263'].str.strip() # Deal with the space
health_19.loc[:, 'healthcare_allo'] = health_19['ch19l263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_19['healthcare_num'] = health_19['ch19l264'].map({"I don't know": 0}).fillna(health_19['ch19l264']).astype(float)
health_19['healthcare_num'] = health_19['healthcare_num'].fillna(0).astype(float)

health_19 = health_19.rename(columns={
    'ch19l001': 'sex',
    'ch19l002': 'age',
    'ch19l016':'height',
    'ch19l017':'weight',

    'ch19l130':'cig_num', 
    'ch19l131':'pipe_num',
    'ch19l132':'cigar_num', 
    'ch19l266':'ecig_num',
    'ch19l133':'drink_freq',
    'ch19l255':'start_date',
    'ch19l256':'start_time',
    'ch19l257':'end_date',
    'ch19l258':'end_time'  
})

health_19_2 = health_19[list_health]


#--------------------------------- 2018 ---------------------------------#

# Data preprocessing: Extract, convert, and clean data structure

health_18['year'] = health_18['ch18k_m'].astype(str).str[:4].astype(int)
health_18.loc[:, 'paid_job'] = health_18['ch18k003'].map({'has paid job': 1}).fillna(0)
health_18.loc[:, 'health'] = health_18['ch18k004'].map({'poor': 0, 'moderate': 1, 'good': 2, 'very good': 3,'excellent': 4})
health_18.loc[:, 'anxiety'] = health_18['ch18k011'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_18.loc[:, 'down'] = health_18['ch18k012'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_18.loc[:, 'calmness'] = health_18['ch18k013'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_18.loc[:, 'depression'] = health_18['ch18k014'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_18.loc[:, 'happiness'] = health_18['ch18k015'].map({'never': 0, 'seldom': 1, 'sometimes': 2, 'often': 3,
'mostly': 4, 'continuously': 5}).fillna(0)
health_18.loc[:, 'chest_pain'] = health_18['ch18k071'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 's_breath'] = health_18['ch18k072'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'coughing'] = health_18['ch18k073'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'headache'] = health_18['ch18k075'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'fatigue'] = health_18['ch18k076'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'sleeping_prb'] = health_18['ch18k077'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'angina'] = health_18['ch18k080'].map({'yes': 1, 'no': 0}).fillna(0)
health_18.loc[:, 'heart_dis'] = health_18['ch18k081'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'hypertension'] = health_18['ch18k082'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'h_cholesterol'] = health_18['ch18k083'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'stroke'] = health_18['ch18k084'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'diabetes'] = health_18['ch18k085'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'cancer'] = health_18['ch18k089'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'benign_tumor'] = health_18['ch18k096'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'hs_smk'] = health_18['ch18k125'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'curr_smk'] = health_18['ch18k126'].map({'yes': 1}).fillna(0)

health_18['ch18k133'] = health_18['ch18k133'].str.strip()
health_18.loc[:, 'drink'] = health_18['ch18k133'].fillna('none')
health_18.loc[:, 'drink'] = health_18['drink'].map({'not at all over the last 12 months': 0, 'none': 0}).fillna(1)
health_18.loc[:, 'drink_7day'] = health_18['ch18k134'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_h_chol'] = health_18['ch18k169'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_hypertension'] = health_18['ch18k170'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_MI'] = health_18['ch18k171'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_heart_dis'] = health_18['ch18k172'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_asthma'] = health_18['ch18k173'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_diabetes'] = health_18['ch18k174'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_joint_pain'] = health_18['ch18k175'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_pains'] = health_18['ch18k176'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_sleeping_prb'] = health_18['ch18k177'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'med_anxiety'] = health_18['ch18k178'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'no_med'] = health_18['ch18k184'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'hospital'] = health_18['ch18k229'].map({'yes': 1}).fillna(0)
health_18['LOS'] = health_18['ch18k230'].fillna(0)
health_18.loc[:, 'operation'] = health_18['ch18k231'].map({'yes': 1}).fillna(0)
health_18.loc[:, 'insur'] = health_18['ch18k239'].map({'yes': 1}).fillna(0)
health_18['obli_risk'] = health_18['ch18k260'].str.extract('(\d+)').fillna(0).astype(int)
health_18['ch18k263'] = health_18['ch18k263'].str.strip() # Deal with the space
health_18.loc[:, 'healthcare_allo'] = health_18['ch18k263'].map({'yes, and the application was adjudged': 1,
'yes, the application is still pending': 1}).fillna(0)
health_18['healthcare_num'] = health_18['ch18k264'].map({"I don't know": 0}).fillna(health_18['ch18k264']).astype(float)
health_18['healthcare_num'] = health_18['healthcare_num'].fillna(0).astype(float)

health_18 = health_18.rename(columns={
    'ch18k001': 'sex',
    'ch18k002': 'age',
    'ch18k016':'height',
    'ch18k017':'weight',

    'ch18k130':'cig_num', 
    'ch18k131':'pipe_num',
    'ch18k132':'cigar_num', 
    'ch18k266':'ecig_num',
    'ch18k133':'drink_freq',
    'ch18k255':'start_date',
    'ch18k256':'start_time',
    'ch18k257':'end_date',
    'ch18k258':'end_time'  
})

health_18_2 = health_18[list_health]


# Merge health data across all years

merged_health = pd.concat([health_23_2,health_22_2, health_21_2, health_20_2, health_19_2,health_18_2], ignore_index=True)
merged_health['year'] = merged_health['year'].astype(int)


# Sort by year
merged_health = merged_health.sort_values(by=['nomem_encr', 'year'])

# Definition of hypertensive patients: self-reported hypertension or having antihypertensive medication
merged_health['hypertension'] = np.where(
    (merged_health['hypertension'] == 1) | (merged_health['med_hypertension'] == 1), 
    1, 
    merged_health['hypertension']
)

# Number of visits of patients
id_counts = merged_health['nomem_encr'].value_counts()
frequency_counts = id_counts.value_counts().sort_index()
merged_health['visit_times'] = merged_health['nomem_encr'].map(id_counts)
# The latest record
merged_health = merged_health.sort_values(by=['nomem_encr', 'hypertension','year'], ascending=[True,True,True])
# Remove duplicates
merged_health_unique = merged_health.drop_duplicates(subset=['nomem_encr'], keep='last') # 9482

#--------------------------------- Background ---------------------------------#
# Row data
background = read_data('background', 'avars_202401_EN_1.0p')

# Data preprocessing: Extract, convert, and clean data structure
background = background.rename(columns={
    'geslacht':'bg_sex',
    'gebjaar': 'birth_year',
    'burgstat': 'marriage',
    'woonvorm': 'Domestic_situation',
    'woning': 'house_status',
    'positie': 'position',
    'sted': 'urban',
    'belbezig': 'pri_occuption',
    'brutoink_f': 'bg_gross_impu_income', # Gross income
    'nettoink_f': 'bg_net_impu_income', # Net income
    'brutocat': 'bg_gross_income_cl',
    'nettocat': 'bg_net_income_cl',
    'oplmet': 'edu1', # Highest level of education with diploma
    'oplcat': 'edu2', # Level of education in CBS (Statistics Netherlands) categories
    'werving': 'ori_wave'
})

list_bg = ['nomem_encr',
           'bg_sex',
           'birth_year',
           'marriage',
           'Domestic_situation',
           'house_status',
           'position',
           'urban',
           'pri_occuption',
           'bg_gross_impu_income',
           'bg_net_impu_income',
           'bg_gross_income_cl',
           'bg_net_income_cl',
           'edu1',
           'edu2',
           'ori_wave'
           ]

background = background[list_bg]

# Merge health with background information
background['bg_info'] = 1
result = merged_health_unique.merge(background, on='nomem_encr', how='left')
result = result[result['bg_info'] == 1]

#--------------------------------- Income ---------------------------------#
# Row data
income_23 = read_data('income', 'ci23p_EN_1.0p')
income_22 = read_data('income', 'ci22o_EN_1.0p')
income_21 = read_data('income', 'ci21n_EN_1.0p')
income_20 = read_data('income', 'ci20m_EN_1.0p')
income_19 = read_data('income', 'ci19l_EN_2.0p')
income_18 = read_data('income', 'ci18k_EN_2.0p')

#--------------------------------- 2023 ---------------------------------#

income_23['year'] = income_23['ci23p_m'].astype(str).str[:4].astype(int) # Income from last year

income_23 = income_23.rename(columns={
    'ci23p005': 'life_satisfaction',
    'ci23p006':'financial_satisfaction'
})

list_income = ['nomem_encr',
               'year',
               'life_satisfaction',
               'financial_satisfaction'
               ]

income_23_2 = income_23[list_income]

#--------------------------------- 2022 ---------------------------------#

income_22['year'] = income_22['ci22o_m'].astype(str).str[:4].astype(int)

income_22 = income_22.rename(columns={
    'ci22o005': 'life_satisfaction',
    'ci22o006':'financial_satisfaction'
})

income_22_2 = income_22[list_income]

#--------------------------------- 2021 ---------------------------------#

income_21['year'] = income_21['ci21n_m'].astype(str).str[:4].astype(int)

income_21 = income_21.rename(columns={
    'ci21n005': 'life_satisfaction',
    'ci21n006':'financial_statistician'
})

income_21_2 = income_21[list_income]


#--------------------------------- 2020 ---------------------------------#

income_20['year'] = income_20['ci20m_m'].astype(str).str[:4].astype(int)

income_20 = income_20.rename(columns={
    'ci20m005': 'life_satisfaction',
    'ci20m006':'financial_statistician'
})

income_20_2 = income_20[list_income]


#--------------------------------- 2019 ---------------------------------#

income_19['year'] = income_19['ci19l_m'].astype(str).str[:4].astype(int)
income_19 = income_19.rename(columns={
    'ci19l005': 'life_satisfaction',
    'ci19l006':'financial_statistician'
})

income_19_2 = income_19[list_income]


#--------------------------------- 2018 ---------------------------------#

income_18['year'] = income_18['ci18k_m'].astype(str).str[:4].astype(int)

income_18 = income_18.rename(columns={
    'ci18k005': 'life_satisfaction',
    'ci18k006':'financial_statistician'
})

income_18_2 = income_18[list_income]


# Merge income data across all years
merged_income = pd.concat([income_23_2, income_22_2, income_21_2, income_20_2, income_19_2,income_18_2], ignore_index=True)

# Merge life satisfaction
merged_income = merged_income.sort_values(by=['nomem_encr', 'year'])
merged_income_unique = merged_income.drop_duplicates(subset=['nomem_encr'], keep='last')

# Merge health&background with life satisfaction
result_2 = result.merge(merged_income_unique, on=['nomem_encr'], how='left')
result_2['income_info'].value_counts()


#--------------------------------- Housing ---------------------------------#
#Row data
housing_23 = read_data('housing', 'cd23p_EN_1.0p')
housing_22 = read_data('housing', 'cd22o_EN_1.0p')
housing_21 = read_data('housing', 'cd21n_EN_1.0p')
housing_20 = read_data('housing', 'cd20m_EN_2.0p')
housing_19 = read_data('housing', 'cd19l_EN_3.0p')
housing_18 = read_data('housing', 'cd18k_EN_1.0p')

#--------------------------------- 2023 ---------------------------------#

for column in housing_23.columns:
    value_counts_with_na = housing_23[column].value_counts(dropna=False)
    print(f"\nValue counts for column '{column}':")
    print(value_counts_with_na)

# Rename
housing_23['year'] = housing_23['cd23p_m'].astype(str).str[:4].astype(int) - 1 # Income for last year
housing_23['cd23p003'] = housing_23['cd23p003'].str.strip()
housing_23['house_status'] = housing_23['cd23p003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_23 = housing_23.rename(columns={
    'cd23p008':'rent_cost', 
    'cd23p010': 'rent_cost_2',
    'cd23p014':'purchase_year',
    'cd23p015':'is_housing_debt', 
    'cd23p083':'remaining_housing_debt',   
    'cd23p025':'house_price', 
    'cd23p026':'house_currency',
    'cd23p018':'is_debt_paid',
    'cd23p019':'debt_num', 
    'cd23p021':'debt_interest_num',
     
    'cd23p078':'start_date',
    'cd23p079':'start_time',
    'cd23p080':'end_date',
    'cd23p081':'end_time' 
})

list_housing = ['nomem_encr',
           'year',
           'house_status',
           'rent_cost', 
           'rent_cost_2',
           'purchase_year', 
           'is_housing_debt',
           'remaining_housing_debt', 
           'house_price', 
           'house_currency',
           'is_debt_paid',
           'debt_num',
           'debt_interest_num',
           'start_date',
           'start_time',
           'end_date',
           'end_time' 
           ]

housing_23_2 = housing_23[list_housing]

#--------------------------------- 2022 ---------------------------------#

# Rename
housing_22 = housing_22.rename(columns={
    'cd22o008':'rent_cost', 
    'cd22o010': 'rent_cost_2',
    'cd22o014':'purchase_year', 
    'cd22o015':'is_housing_debt', 
    'cd22o083':'remaining_housing_debt',   
    'cd22o025':'house_price', 
    'cd22o026':'house_currency',
    'cd22o018':'is_debt_paid', 
    'cd22o019':'debt_num',
    'cd22o021':'debt_interest_num',
     
    'cd22o078':'start_date',
    'cd22o079':'start_time',
    'cd22o080':'end_date',
    'cd22o081':'end_time' 
})

housing_22['year'] = housing_22['cd22o_m'].astype(str).str[:4].astype(int) - 1 
housing_22['cd22o003'] = housing_22['cd22o003'].str.strip()
housing_22['house_status'] = housing_22['cd22o003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_22_2 = housing_22[list_housing]

#--------------------------------- 2021 ---------------------------------#

# Rename
housing_21 = housing_21.rename(columns={
    'cd21n008':'rent_cost', 
    'cd21n010': 'rent_cost_2',
    'cd21n014':'purchase_year', 
    'cd21n015':'is_housing_debt', 
    'cd21n083':'remaining_housing_debt',   
    'cd21n025':'house_price', 
    'cd21n026':'house_currency',
    'cd21n018':'is_debt_paid',
    'cd21n019':'debt_num', 
    'cd21n021':'debt_interest_num',
     
    'cd21n078':'start_date',
    'cd21n079':'start_time',
    'cd21n080':'end_date',
    'cd21n081':'end_time' 
})

housing_21['year'] = housing_21['cd21n_m'].astype(str).str[:4].astype(int) - 1
housing_21['cd21n003'] = housing_21['cd21n003'].str.strip()
housing_21['house_status'] = housing_21['cd21n003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_21_2 = housing_21[list_housing]

#--------------------------------- 2020 ---------------------------------#

# Rename
housing_20 = housing_20.rename(columns={
    'cd20m008':'rent_cost', 
    'cd20m010': 'rent_cost_2',
    'cd20m014':'purchase_year', 
    'cd20m015':'is_housing_debt', 
    'cd20m083':'remaining_housing_debt',   
    'cd20m025':'house_price', 
    'cd20m026':'house_currency',
    'cd20m018':'is_debt_paid', 
    'cd20m019':'debt_num',
    'cd20m021':'debt_interest_num',
     
    'cd20m078':'start_date',
    'cd20m079':'start_time',
    'cd20m080':'end_date',
    'cd20m081':'end_time' 
})

housing_20['year'] = housing_20['cd20m_m'].astype(str).str[:4].astype(int) - 1 
housing_20['cd20m003'] = housing_20['cd20m003'].str.strip()
housing_20['house_status'] = housing_20['cd20m003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_20_2 = housing_20[list_housing]

#--------------------------------- 2019 ---------------------------------#

# Rename
housing_19 = housing_19.rename(columns={
    'cd19l008':'rent_cost', 
    'cd19l010': 'rent_cost_2',
    'cd19l014':'purchase_year', 
    'cd19l015':'is_housing_debt', 
    'cd19l083':'remaining_housing_debt',   
    'cd19l025':'house_price', 
    'cd19l026':'house_currency',
    'cd19l018':'is_debt_paid', 
    'cd19l019':'debt_num', 
    'cd19l021':'debt_interest_num',
    'cd19l078':'start_date',
    'cd19l079':'start_time',
    'cd19l080':'end_date',
    'cd19l081':'end_time' 
})

housing_19['year'] = housing_19['cd19l_m'].astype(str).str[:4].astype(int) - 1 
housing_19['cd19l003'] = housing_19['cd19l003'].str.strip()
housing_19['house_status'] = housing_19['cd19l003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_19_2 = housing_19[list_housing]

#--------------------------------- 2018 ---------------------------------#

# Rename
housing_18 = housing_18.rename(columns={
    'cd18k008':'rent_cost', 
    'cd18k010': 'rent_cost_2',
    'cd18k014':'purchase_year', 
    'cd18k015':'is_housing_debt', 
    'cd18k083':'remaining_housing_debt',   
    'cd18k025':'house_price', 
    'cd18k026':'house_currency',
    'cd18k018':'is_debt_paid', 
    'cd18k019':'debt_num', 
    'cd18k021':'debt_interest_num',
     
    'cd18k078':'start_date',
    'cd18k079':'start_time',
    'cd18k080':'end_date',
    'cd18k081':'end_time' 
})

housing_18['year'] = housing_18['cd18k_m'].astype(str).str[:4].astype(int) - 1
housing_18['cd18k003'] = housing_18['cd18k003'].str.strip()
housing_18['house_status'] = housing_18['cd18k003'].map({
    '(co-)owner': 'owner',
    'tenant': 'tenant',
    'subtenant': 'tenant'
}).fillna('other')

housing_18_2 = housing_18[list_housing]

# Merge housing data across all years
merged_housing = pd.concat([housing_23_2, housing_22_2, housing_21_2, housing_20_2, housing_19_2,housing_18_2], ignore_index=True)


# Sort by year
merged_housing = merged_housing.sort_values(by=['nomem_encr', 'year'])
merged_housing = merged_housing.drop(columns=['start_date', 'start_time', 'end_date','end_time'])
merged_housing['housing_info'] = 1
# Remove duplicates
merged_housing_unique = merged_housing.drop_duplicates(subset=['nomem_encr'], keep='last')
# Merge with results_2 with housing data
result_3 = result_2.merge(merged_housing_unique, on=['nomem_encr'], how='left')
result_3['housing_info'].value_counts()

#--------------------------------- Asset ---------------------------------#
# Row data
assets_22 = read_data('assets', 'ca22h_EN_1.0p') 
assets_20 = read_data('assets', 'ca20g_EN_1.0p')
assets_18 = read_data('assets', 'ca18f_EN_1.0p')
assets_16 = read_data('assets', 'ca16e_EN_1.0p')

#--------------------------------- 2022 ---------------------------------#

assets_22 = assets_22.rename(columns={
    'ca22h012':'bank_balance', 
    'ca22h018':'real_estate', 
    'ca22h023':'car_boat',   
    'ca22h026':'loaned_out_money', 
    'ca22h027':'other_assets',
    'ca22h063':'credit_debt'
})

assets_22['year'] = assets_22['ca22h_m'].astype(str).str[:4].astype(int)

list_assets = ['nomem_encr',
           'year',
           'bank_balance',
           'real_estate',
           'car_boat', 
           'other_assets', 
           'credit_debt'
           ]

assets_22_2 = assets_22[list_assets]

#--------------------------------- 2020 ---------------------------------#

assets_20 = assets_20.rename(columns={
    'ca20g012':'bank_balance', 
    'ca20g018':'real_estate',
    'ca20g023':'car_boat', 
    'ca20g027':'other_assets', 
    'ca20g063':'credit_debt'
})

assets_20['year'] = assets_20['ca20g_m'].astype(str).str[:4].astype(int)
assets_20_2 = assets_20[list_assets]

#--------------------------------- 2018 ---------------------------------#

assets_18= assets_18.rename(columns={
    'ca18f012':'bank_balance', 
    'ca18f018':'real_estate', 
    'ca18f023':'car_boat', 
    'ca18f027':'other_assets', 
    'ca18f063':'credit_debt'
})
assets_18['year'] = assets_18['ca18f_m'].astype(str).str[:4].astype(int)
assets_18_2 = assets_18[list_assets]

merged_assets = pd.concat([assets_22_2, assets_20_2, assets_18_2], ignore_index=True)

# Sort by year
merged_assets = merged_assets.sort_values(by=['nomem_encr', 'year'])
merged_assets['assets_info'] = 1
# Remove duplicates
merged_assets_unique = merged_assets.drop_duplicates(subset=['nomem_encr'], keep='last')
# Merge result_3 with Asset data
result_4 = result_3.merge(merged_assets_unique, on='nomem_encr', how='left', suffixes=('', '_drop'))
result_4.drop([col for col in result_4.columns if 'drop' in col], axis=1, inplace=True)
result_4['assets_info'].value_counts()


# Remove records with missing values
imputed_result = result_4.copy()

# Exclude patients with missing housing data
imputed_result = imputed_result[imputed_result['housing_info'] == 1]
# Exclude patients with missing asset data
imputed_result = imputed_result[imputed_result['assets_info'] == 1]
# Exclude patients with missing income data
imputed_result = imputed_result[imputed_result['bg_net_impu_income'].notna()]
# Exclude patients with missing life satisfaction
imputed_result = imputed_result[imputed_result['life_satisfaction'].notna()]

# Age
# Impute missing 'age' using current year - birth year
imputed_result['age_2'] = imputed_result['year'] - imputed_result['birth_year']
imputed_result['age'] = imputed_result['age'].fillna(imputed_result['age_2'])
imputed_result.drop(columns=['age_2'], inplace=True)
imputed_result.describe(imputed_result['age'])
imputed_result['age'].describe()
imputed_result['age_5'] = imputed_result['age']/5

# Density plot of age
imputed_result['yearly_net_income_k'] = imputed_result['yearly_net_income'] / 1000
filtered_income = imputed_result[imputed_result['yearly_net_income_k'] < 200]

plt.hist(imputed_result['age'], bins=30, color='blue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Style of Seaborn
sns.set(style="whitegrid")

sns.histplot(data=imputed_result, x='age', bins=30, kde=False, color='blue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Classify age into 5 groups
def classify_age(row):
    if 0 < row['age'] < 45:
        return '1'
    elif 45 <= row['age'] < 55:
        return '2'
    elif 55 <= row['age'] < 65:
        return '3'
    elif 65 <= row['age'] < 75:
        return '4'
    elif row['age'] >= 75:
        return '5'
    
imputed_result['age_group'] = imputed_result.apply(classify_age, axis=1)
imputed_result['age_group'] = imputed_result['age_group'].astype(int)


# Exclude patients whose sex = Other 
imputed_result['bg_sex'] = imputed_result['bg_sex'].str.strip()
imputed_result = imputed_result[imputed_result['bg_sex'] != 'Other']
imputed_result['male'] = np.where(imputed_result['bg_sex'] == 'Male', 1, 0)

# Exclude patients with missing BMI
imputed_result['height'] = imputed_result['height'].apply(lambda x: x + 100 if x < 100 else x)
median_weight = imputed_result['weight'].median()
imputed_result['weight'] = imputed_result['weight'].apply(lambda x: median_weight if x < 20 else x)
imputed_result['BMI'] = imputed_result['weight'] / ((imputed_result['height']/100) ** 2).round(2)
imputed_result = imputed_result[(imputed_result['BMI'].notna())] 

# Classify BMI into 4 groups
def classify_BMI(row):
    if row['BMI'] < 18.5:
        return 'Underweight'
    elif 18.5 <= row['BMI'] < 25:
        return 'Normal'
    elif 25 <= row['BMI'] < 30:
        return 'Overweight'
    else:
        return 'Obesity'

imputed_result['BMI_cls'] = imputed_result.apply(classify_BMI, axis=1)

# Classified education
imputed_result['edu1'] = imputed_result['edu1'].str.strip()
imputed_result.loc[:, 'edu_cls'] = imputed_result['edu1'].map({
    'Not (yet) completed any education': 1, 
    'Not yet started any education*': 1, 
    'other': 1, 
    'primary school': 1,
    'vmbo (intermediate secondary education, US: junior high school)': 2,
    'havo/vwo (higher secondary education/preparatory university education, US: senio': 3,
    'mbo (intermediate vocational education, US: junior college)': 4,
    'hbo (higher vocational education, US: college)': 5,
    'wo (university)': 6
})


# Has job
imputed_result.loc[:, 'pri_occuption_x'] = imputed_result['pri_occuption'].map({
    'Paid employment': 'Paid employment', 
    'Is pensioner ([voluntary] early retirement, old age pension scheme)': 'Is pensioner'
}).fillna('No employment')

# Marriage
imputed_result.loc[:, 'marriage_x'] = imputed_result['marriage'].map({
    'Married': 'Married', 
    'Never been married': 'Never_been_married',
    'Divorced': 'Divorced_separated',
    'Separated': 'Divorced_separated',
    'Widow or widower': 'Widow_widower'
})

# Depression and anxiety

def classify_depression(row):
    if row['med_anxiety'] == 1:
        return 3
    elif (row['anxiety'] in [3, 4, 5]) or (row['depression'] in [3, 4, 5]):
        return 3
    elif (row['anxiety'] in [1, 2]) or (row['depression'] in [1, 2]):
        return 2
    return 1

imputed_result['depression_y'] = imputed_result.apply(classify_depression, axis=1)


# Sleep problem

imputed_result['sleeping_prb'] = np.where(
    (imputed_result['sleeping_prb'] == 1) | (imputed_result['med_sleeping_prb'] == 1), 
    1, 
    imputed_result['sleeping_prb']
)

# Smoking status: smoke in current and in the past
imputed_result['smk'] = np.where((imputed_result['curr_smk'] == 1) | (imputed_result['hs_smk'] == 1),1,0)

def classify_smk(row):
    if row['curr_smk'] == 1:
        return 'cur_smk'
    elif row['hs_smk'] == 1:
        return 'hs_smk'
    else:
        return 'never_smk'

imputed_result['smk_cls'] = imputed_result.apply(classify_smk, axis=1)


# Assets: asset_total = bank_balance + real_estate + car or boats + other property
imputed_result['bank_balance_x'] = imputed_result['bank_balance'].astype(float)
# Revise the value < 0 or >99999999
imputed_result['bank_balance_x'] = imputed_result['bank_balance_x'].apply(lambda x: 0 if (x < 0 or x > 99999999) else x)

# Remove strings such as "I don't know" & "I prefer not to say"
def remove_string(value):
    if isinstance(value, str):  # Ensure the value is a string
        has_digit = any(char.isdigit() for char in value)  # Check for digits
        has_alpha = any(char.isalpha() for char in value)  # Check for alphabets
        if has_digit and has_alpha:
            return value  # Keep the original string
        elif has_digit:
            return value  # Keep numbers
        else:
            return np.nan  # Convert strings with only letters to NaN
    return value

# Apply the function conditionally based on 'is_housing_debt' column

imputed_result['real_estate_x'] = imputed_result['real_estate'].apply(remove_string)
imputed_result['real_estate_x'] = imputed_result['real_estate_x'].astype(float)
imputed_result['real_estate_x'] = imputed_result['real_estate_x'].apply(lambda x: 0 if (x < 0 or x > 99999999) else x)
imputed_result['car_boat_x'] = imputed_result['car_boat'].apply(remove_string)
imputed_result['car_boat_x'] = imputed_result['car_boat_x'].astype(float)
imputed_result['car_boat_x'] = imputed_result['car_boat_x'].apply(lambda x: 0 if (x < 0 or x > 99999999) else x)
imputed_result['other_assets_x'] = imputed_result['other_assets'].apply(remove_string)
imputed_result['other_assets_x'] = imputed_result['other_assets_x'].astype(float)
imputed_result['other_assets_x'] = imputed_result['other_assets_x'].apply(lambda x: 0 if (x < 0 or x > 99999999) else x)

imputed_result['asset_total'] = imputed_result[['bank_balance_x', 'real_estate_x', 'car_boat_x', 'other_assets_x']].sum(axis=1, skipna=True)


# Housing debts
imputed_result['remaining_housing_debt_x'] = imputed_result.apply(
    lambda row: remove_string(row['remaining_housing_debt']) if row['is_housing_debt'] == 'yes' else row['remaining_housing_debt'],
    axis=1
)
imputed_result['remaining_housing_debt_x'] = imputed_result['remaining_housing_debt_x'].astype(float)

# Convert guilder to euros
imputed_result['remaining_housing_debt_x'] = imputed_result.apply(
    lambda row: row['remaining_housing_debt_x'] * 0.45378022 if row['house_currency'] == 'guilder' 
    else row['remaining_housing_debt_x'],
    axis=1
) 
# Correct abnormal values
imputed_result['remaining_housing_debt_x'] = imputed_result['remaining_housing_debt_x'].apply(
    lambda x: np.nan if x >= 999999999 else x
)  

# Define the household without house debt
imputed_result['is_housing_debt'] = imputed_result['is_housing_debt'].fillna('no')
imputed_result.loc[
    (imputed_result['remaining_housing_debt_x'].isna()) & (imputed_result['is_housing_debt'] == 'no'),
    'remaining_housing_debt_x'
] = 0 

# Exclude "I don't know" | "prefer not to say"
imputed_result = imputed_result.dropna(subset=['remaining_housing_debt_x'])

# All debt：housing debt + credit_debt
imputed_result['credit_debt_x'] = imputed_result['credit_debt'].apply(remove_string)
imputed_result['credit_debt_x'] = imputed_result['credit_debt_x'].astype(float)
imputed_result['credit_debt_x'] = imputed_result['credit_debt_x'].apply(lambda x: 0 if (x < 0 or x > 99999999) else x)

imputed_result['debt_total'] = imputed_result[['remaining_housing_debt_x', 'credit_debt_x']].sum(axis=1, skipna=True)


# Distribution of Asset
pd.options.display.float_format = '{:.2f}'.format
imputed_result[['asset_total']].describe()
# Scatter plot of Asset
plt.figure(figsize=(10, 6))
sns.scatterplot(data=imputed_result, x=imputed_result.index, y='asset_total', palette='coolwarm', s=100)
plt.xlabel('Index')
plt.ylabel('Assets')
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Yearly_net_income
imputed_result['yearly_net_income'] = imputed_result['bg_net_impu_income'] * 12
pd.options.display.float_format = '{:.2f}'.format
imputed_result['yearly_net_income'].describe()

# Scatter plot of yearly_net_income

# # Create a function to format numbers with thousand separators for better readability
def thousands_formatter(x, pos):
    return '{:,.0f}'.format(x)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=imputed_result, x=imputed_result.index, y='yearly_net_income', s=100)
plt.ylabel('Yearly Net Income')

plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.grid(True)
plt.xticks([]) 
plt.xlabel('')  
plt.show()


# classify net incomes based on the p25,median,p75

def classify_net_income3(row):
    if 0 <= row['yearly_net_income'] < 17619:
        return 1
    elif 17619 <= row['yearly_net_income'] < 27000:
        return 2
    elif 27000 <= row['yearly_net_income'] < 36000:
        return 3
    elif row['yearly_net_income'] >= 36000:
        return 4
    
imputed_result['net_income_cls3'] = imputed_result.apply(classify_net_income3, axis=1)

category_mapping = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very_High'
}

imputed_result['net_income_cls3_c'] = imputed_result['net_income_cls3'].map(category_mapping)
net_income_dummies = pd.get_dummies(imputed_result['net_income_cls3_c'], prefix='inc_cls3')
imputed_result = pd.concat([imputed_result, net_income_dummies], axis=1)

def classify_asset_cls(row):
    if 0 <= row['asset_total'] < 9000:
        return 1
    elif 9000 <= row['asset_total'] < 41000:
        return 2
    elif row['asset_total'] >= 41000:
        return 3
    
imputed_result['asset_total_cls'] = imputed_result.apply(classify_asset_cls, axis=1)

category_mapping = {
    1: 'Low',
    2: 'Medium',
    3: 'High'
}

imputed_result['asset_total_cls_c'] = imputed_result['asset_total_cls'].map(category_mapping)
net_income_dummies = pd.get_dummies(imputed_result['asset_total_cls_c'], prefix='asst_cls')
imputed_result = pd.concat([imputed_result, net_income_dummies], axis=1)


#  classify debts based on none, the p25,median,p75

def classify_debt4(row):
    if row['debt_total'] == 0:
        return 1
    elif row['debt_total'] < 57934:
        return 2
    elif 57934 <= row['debt_total'] < 135200:
        return 3
    elif 135200 <= row['debt_total'] < 229000:
        return 4
    elif row['debt_total'] >= 229000:
        return 5
    
imputed_result['debt_total_cls2'] = imputed_result.apply(classify_debt4, axis=1)

category_mapping = {
    1: 'None',
    2: 'Low',
    3: 'Medium',
    4: 'High',
    5: 'Very_High'
}

imputed_result['debt_total_cls2_c'] = imputed_result['debt_total_cls2'].map(category_mapping)
net_income_dummies = pd.get_dummies(imputed_result['debt_total_cls2_c'], prefix='db_ttl2_cls')
imputed_result = pd.concat([imputed_result, net_income_dummies], axis=1)

# One hot 
marriage_dummies = pd.get_dummies(imputed_result['marriage_x'], prefix='marr')
imputed_result = pd.concat([imputed_result, marriage_dummies], axis=1)

occuption_dummies = pd.get_dummies(imputed_result['pri_occuption_x'], prefix='occu')
imputed_result = pd.concat([imputed_result, occuption_dummies], axis=1)

imputed_result['househead'] = (imputed_result['position'] == 'Household head')
imputed_result['age_45'] = imputed_result['age'] >= 45
imputed_result['age_65'] = imputed_result['age'] >= 65
imputed_result['high_debt2'] = (imputed_result['debt_total'] >= 135200) # Median
imputed_result['no_debt'] = (imputed_result['debt_total'] == 0)


# Define life satisfaction to 
def map_to_5_scale(x):
    if x <= 2:
        return 1
    elif x <= 4:
        return 2
    elif x <= 6:
        return 3
    elif x <= 8:
        return 4
    else:
        return 5

imputed_result['life_satisfaction_x'] = imputed_result['life_satisfaction'].apply(map_to_5_scale)
