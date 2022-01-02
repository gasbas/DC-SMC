import pandas as pd
import numpy as np
import json

def process_data(load_path, replacements_path = 'replacements.json', year = '') : 
    
    df  = pd.read_excel(load_path,sheet_name=2, header = 3)
    df_copy = df.iloc[np.where(df['% 3+'].values != 'n<25')].copy()
    df_copy = df_copy.iloc[np.where(df_copy['% 3+'].values != 'n<10')]
    
    #Replacing school names so they match between all datasets
    if year == '2017' : 
        school_names = np.array([i.replace('@','at').replace('ES', 'Elementary School').replace('EC','Education Campus').replace('PCS ','PCS - ').replace('KIPP DC','KIPP DC -').split(' at ')[0] for i in df_copy['School Name']])

        with open(replacements_path, 'r') as f : 
            replacements = json.load(f)
        ids, new_names = [], []
        for old_sn,new_name in replacements.items() : 
            try : 
                ids.append(np.where(school_names == old_sn)[0][0])
                new_names.append(new_name)
            except :  
                pass #no matching ids
        school_names[ids] = new_names
    else : 
        school_names = [i.split(' @ ')[0] for i in df_copy['School Name']]
        
    df_copy['School Name'] = school_names    
    
    df_copy = df_copy[['School Ward', 'LEA Name', 'School Name','% 3+', 'Total  Valid Test Takers']]
    target = [float(i[:4])/100 for i in df_copy['% 3+'].values]
    df_copy.rename(columns={'Total  Valid Test Takers':'trials'},inplace=True)
    df_copy['trials'] = df_copy['trials'].astype(int)
    df_copy['successes'] = (target * df_copy['trials']).astype(int)
    df_copy['year'] = year
    return df_copy.reset_index(drop=True).drop('% 3+',axis=1)

if __name__ == '__main__' : 

    df = pd.DataFrame()
    school_names = {}
    for year in ['2017','2018','2019'] : 
        tmp = process_data(f'{year}_raw_data.xlsx', year = year)
        df = df.append(tmp)
        school_names[year] = tmp['School Name'].values
        
    valid_schools = [i for i in school_names['2017'] if i in school_names['2018'] and i in school_names['2019']]

    df = df[df['School Name'].isin(valid_schools)]

    #Encode school, LEA and ward names into integers 

    def label_encode(data_to_encode) : 
        return {val : s_id for s_id,val in zip([i for i in range(len(data_to_encode))], data_to_encode)}

    school_ids = label_encode(df['School Name'].unique())
    LEA_ids = label_encode(df['LEA Name'].unique())
    ward_ids = label_encode(df['School Ward'].unique())

    df['school_id'] = df['School Name'].apply(lambda x: school_ids[x])
    df['lea_id'] = df['LEA Name'].apply(lambda x: LEA_ids[x])
    df['ward_id'] = df['School Ward'].apply(lambda x: ward_ids[x])

    df.drop(['School Name', 'LEA Name', 'School Ward'],axis=1,inplace=True)

    #Saving final dataframe

    final_df = df.sort_values(['school_id','lea_id','ward_id','year'])

    final_df.to_csv('preprocessed_DC_data.csv',index=False)

    #Saving label encondigs to json files for later use
    with open('schools.json','w') as f :
        json.dump(school_ids,f)
    with open('LEA.json','w') as f :
        json.dump(LEA_ids,f)
    with open('wards.json','w') as f :
        json.dump(ward_ids,f)