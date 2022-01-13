import pandas as pd
import numpy as np
import csv
import pycountry
from pycountry_convert import country_alpha3_to_country_alpha2, country_alpha2_to_country_name


COUNTRIES = ['Nigéria', 'Nouvelle_Zélande', 'États-Unis', 'Espagne', 'Canada',
             'Grèce', 'United_Kingdom', 'Israël', 'France', 'Bahamas',
             'Lettonie', 'Géorgie', 'Serbie', 'Croatie', 'Soudan', 'Argentine',
             'Allemagne', 'Saint_Lucia', 'Slovénie', 'Suisse', 'Sénégal',
             'République_dominicaine', 'Cameroun', 'Angola', 'Turquie', 'Italie',
             'Australie', 'Japan', 'Republic_of_the_Congo', 'République_tchèque',
             'Lituanie','République_Démocratique_du_Congo', 'Ukraine',
             'Brésil', 'Finland', 'Egypt', 'Bosnia_and_Herzegovina',
             'Austria', 'Portugal', 'Jamaïque', 'Monténégro']
CODES = ['NGA','NZL','USA','ESP','CAN','GRC','GBR','ISR','FRA','BHS','LVA','GEO',
         'SRB','HRV','SDN','ARG','DEU','LCA','SVN','CHE','SEN','DOM','CMR','AGO',
         'TUR','ITA','AUS','JPN','COD','CZE','LTU','COD','UKR','BRA','FIN','EGY',
         'BIH','AUT','PRT','JAM','MNE']
DICT_ALPHA3 = {COUNTRIES[i] : CODES[i] for i in range(len(COUNTRIES))}



#################################################################################################################################
#                                                                                                                               #
#                                                           PROCESSING                                                          #
#                                                                                                                               #
#################################################################################################################################
def remove_nan_lines(df):
    '''Retire les lignes dans lesquelles il y a une valeur Nan'''
    return df.dropna()


def remove_nan_columns(df):
    '''Retire les colonnes dans lesquelles il y a une valeur Nan'''
    return df.dropna(axis=1)


def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            code=DICT_ALPHA3[country]
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE


def process_data_players(df):
    '''Processing du dataset des stats de joueurs'''
    # Remove Nan
    print(df.shape)
    df = remove_nan_lines(df)
    print(df.shape)

    # Set POSS to float type
    df['POSS'] = df['POSS'].map(lambda x:x.replace(",","."))
    df['POSS'] = df['POSS'].astype('float64')

    # Get alpha3 country codes
    df['iso_alpha'] = alpha3code(df['Pays'])
    df['country'] = df['iso_alpha'].apply(lambda x: country_alpha2_to_country_name(country_alpha3_to_country_alpha2(x)))
    df['PDV'] = df['PDV'].replace('C-F','F-C').replace('F-G','G-F')
    df['PDV'] = df['PDV'].replace('F-C','F').replace('G-F','F')
    
    return df


def process_data_teams(df):
    '''Pour l'instant on met juste les colonnes au bon format'''
    df['POSS'] = df['POSS'].map(lambda x:x.replace(",","."))
    df['POSS'] = df['POSS'].astype('float64')
    return df



#################################################################################################################################
#                                                                                                                               #
#                                                               MERGING                                                         #
#                                                                                                                               #
#################################################################################################################################
def get_common_columns(list_of_df):
    d = [df.columns for df in list_of_df]
    return set(d[0]).intersection(*d[1:])


def merge_all_datasets(list_of_df, key='PLAYER', name='df_players_merged'):
    
    common_cols = [i for i in get_common_columns([list_of_df[0], list_of_df[1]]) if i != key]
    list_of_df[1] = list_of_df[1].drop(list(common_cols), axis=1)
    df_new = list_of_df[0].merge(list_of_df[1], how='left', on=key)
    
    for df in list_of_df[2:]:
        common_cols = [ i for i in get_common_columns([df_new, df]) if i != key]
        df = df.drop(list(common_cols), axis=1)
        df_new = df_new.merge(df, how='left', on=key)
    
        
    # On sauve le dataframe créé
    df_new.to_excel('datasets/'+name+'.xlsx')   # Pour un meilleur apperçu des données
    df_new.to_csv('datasets/'+name+'.csv')
    
    return df_new



#################################################################################################################################
#                                                                                                                               #
#                                                   PROCESS MATCH DATASET                                                       #
#                                                                                                                               #
#################################################################################################################################

def process_match(file):
    DATA=[]
    EQUIPES=[]
    SCORES=[]
    SCOREnbWIN=[]
    ANNE=[]

    dic={
        'Année' : [],
        'Equipes' : [],
        'SCOREnbWIN': []
    }

    f= open(file,'r')
    myReader = csv.reader(f)

    for row in myReader:
        DATA.append(row)

    def Trie(i):
        if DATA[i][2] not in EQUIPES:
            EQUIPES.append(DATA[i][2])
            SCORES.append([])
            if int(DATA[i][4])>int(DATA[i][5]):
                SCORES[len(SCORES)-1].append("W")
            else:
                SCORES[len(SCORES)-1].append("L")
        else:
            if int(DATA[i][4])>int(DATA[i][5]):
                SCORES[EQUIPES.index(DATA[i][2])].append("W")
            else:
                SCORES[EQUIPES.index(DATA[i][2])].append("L")

        if DATA[i][3] not in EQUIPES:
            EQUIPES.append(DATA[i][3])
            SCORES.append([])
            if int(DATA[i][4])<int(DATA[i][5]):
                SCORES[len(SCORES)-1].append("W")
            else:
                SCORES[len(SCORES)-1].append("L")
        else:
            if int(DATA[i][4])<int(DATA[i][5]):
                SCORES[EQUIPES.index(DATA[i][3])].append("W")
            else:
                SCORES[EQUIPES.index(DATA[i][3])].append("L")


    def MakeRatio():
        for i in range (len(SCORES)):
            compteur=0
            for ii in range (len(SCORES[i])):
                if SCORES[i][ii]=='W':
                    compteur+=1

            SCOREnbWIN.append(round(compteur/len(SCORES[i])*100,2))

    ##
    AnneActuel=DATA[1][1][0] + DATA[1][1][1] + DATA[1][1][2] +DATA[1][1][3]
    dic['Année'].append(AnneActuel)
    for i in range (1,len(DATA)):
        if "None" not in DATA[i]:
            if (DATA[i][1][0] + DATA[i][1][1] + DATA[i][1][2] +DATA[i][1][3]!=AnneActuel):
                MakeRatio()

                dic['Année'].append(DATA[i][1][0] + DATA[i][1][1] + DATA[i][1][2] +DATA[i][1][3])
                dic['Equipes'].append(EQUIPES)
                dic['SCOREnbWIN'].append(SCOREnbWIN)


                ANNE.append(DATA[i][1][0] + DATA[i][1][1] + DATA[i][1][2] +DATA[i][1][3])

                EQUIPES=[]
                SCOREnbWIN=[]
                SCORES=[]
                AnneActuel=DATA[i][1][0] + DATA[i][1][1] + DATA[i][1][2] +DATA[i][1][3]


            Trie(i)

    EquipeScoreAnne=[]
    EquipeSauv=[]

    equipeactuel=""

    ar = []

    for i in range (len(dic['Equipes'])):
        for ii in range (len(dic['Equipes'][i])):#dic['Equipes'][i][ii])
            if dic['Equipes'][i][ii] not in EquipeSauv:
                EquipeSauv.append(dic['Equipes'][i][ii])
                equipeactuel=dic['Equipes'][i][ii]

                for iii in range (len(dic['Equipes'])):
                    if equipeactuel in dic['Equipes'][iii]:
                        pos=dic['Equipes'][iii].index(equipeactuel)
                        EquipeScoreAnne.append( dic['SCOREnbWIN'][iii][pos])
                    else:
                        EquipeScoreAnne.append(-1)

                ar.append(EquipeScoreAnne)
                EquipeScoreAnne=[]

    df = pd.DataFrame(ar)
    df = df.transpose()
    df.columns = EquipeSauv

    df.to_csv('datasets/df_matchs.csv')

    return df




#################################################################################################################################
#                                                                                                                               #
#                                                               MAIN                                                            #
#                                                                                                                               #
#################################################################################################################################
if __name__ == '__main__':
    
    ### Merging players datasets ###
    dfp1 = pd.read_csv("datasets/stats_players_trad.csv").drop(["Unnamed: 0"], axis=1)
    dfp2 = pd.read_csv("datasets/stats_players_advanced.csv").drop(["Unnamed: 0"], axis=1)
    dfp3 = pd.read_csv("datasets/stats_players_scoring.csv").drop(["Unnamed: 0"], axis=1)
    dfp4 = pd.read_csv("datasets/stats_players_usage.csv").drop(["Unnamed: 0"], axis=1)

    dfp_a = pd.read_csv("datasets/players.csv").rename({"Unnamed: 0":"PLAYER", "Equipe":"TEAM"}, axis=1).drop(['Unit'], axis=1)
    dfp_a['PLAYER'] = dfp_a['PLAYER'].map(lambda x:x.replace('III',' III').replace('II',' II').replace('IV',' IV').replace('  ',' '))
    dfp_a['PLAYER'] = dfp_a['PLAYER'].map(lambda x:x.replace('Jr.',' Jr.').replace('Sr.',' Sr.').replace('  ',' '))
    dfp_a['PLAYER'] = dfp_a['PLAYER'].map(lambda x:x.replace(" '","'"))

    list_of_df = [dfp1, dfp2, dfp3, dfp4, dfp_a]
    df_to_rule_them_all = merge_all_datasets(list_of_df, key='PLAYER', name='df_players_merged')


    ### Merging teams datasets ###
    dft1 = pd.read_csv("datasets/stats_teams_trad.csv").drop(["Unnamed: 0"], axis=1)
    dft2 = pd.read_csv("datasets/stats_teams_advanced.csv").drop(["Unnamed: 0"], axis=1)

    list_of_df = [dft1, dft2]
    df_to_rule_them_all = merge_all_datasets(list_of_df, key='TEAM', name='df_teams_merged')



    ### Processing data ###
    df = pd.read_csv('datasets/df_players_merged.csv').drop(["Unnamed: 0"], axis=1)
    df = process_data_players(df)
    df.to_excel('datasets/df_players_merged.xlsx')
    df.to_csv('datasets/df_players_merged.csv')

    df = pd.read_csv('datasets/df_teams_merged.csv').drop(["Unnamed: 0"], axis=1)
    df = process_data_teams(df)
    df.to_csv('datasets/df_teams_merged.csv')
    df.to_excel('datasets/df_teams_merged.xlsx')


    ### Processing Matchs data ###
    process_match("datasets/ScoreEquipeNBA.csv")







