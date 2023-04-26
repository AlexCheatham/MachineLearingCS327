import pandas as pd

def read_excel(name, outputName):
    df = pd.read_excel(name)
    clean_data(df, outputName)

def clean_data(df, outputName):
    # Drop unneeded columns
    df = df.drop('Name', axis = 1)
    #df = df.drop('Genre', axis = 1)
    #df = df.drop('Developer', axis = 1)
    #df = df.drop('Rating', axis = 1)
    
    #COMMENT THIS OUT IF USING GAMEDATACLEANED=====================================
    df['Story_Focus'] = df['Story Focus'].apply(lambda x: 1 if x == 'x' else 0)
    df['Gameplay_Focus'] = df['Gameplay Focus'].apply(lambda x: 1 if x == 'x' else 0)
    df['Series_Focus'] = df['Series'].apply(lambda x: 1 if x == 'x' else 0)
    df = df.drop('Story Focus', axis = 1)
    df = df.drop('Gameplay Focus', axis = 1)
    df = df.drop('Series', axis = 1)
    #===========================================================
    df['Published_in_NA'] = df['NA_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_EU'] = df['EU_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_JP'] = df['JP_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_Other'] = df['Other_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('NA_Sales', axis = 1)
    df = df.drop('EU_Sales', axis = 1)
    df = df.drop('JP_Sales', axis = 1)
    df = df.drop('Other_Sales', axis = 1)
    # Scale user score to out of 100
    genres = map_genre(df)
    genre_list = df['Genre'].tolist()
    for i in range(len(genre_list)):
        genre_list[i] = genres[genre_list[i]]
    
    df['Genre'] = genre_list
    
    developers = map_developer(df)
    developer_list = df['Developer'].tolist()
    for i in range(len(developer_list)):
        developer_list[i] = developers[developer_list[i]]
    
    df['Developer'] = developer_list
    
    publishers = map_publisher(df)
    
    publisher_list = df['Publisher'].tolist()
    for i in range(len(publisher_list)):
        publisher_list[i] = publishers[publisher_list[i]]
    
    df['Publisher'] = publisher_list
    df['User_Score'] = df['User_Score']*10
    
    df['Rating'] = map_generic(df, 'Rating')
    #read mapped file and replace publisher with their id. You may want to first convert the publisher into lower case and then 
    #write the id.
    # Saves dataframe to excel spreadsheet
    df.to_excel(outputName)

def map_publisher(df):
    #plug in the csv creation and header part here, publisher and corresponding ID, here you will have a csv writer object obj
    #df=pd.read_excel(CleanedData2.xlsx)
    publisher=df['Publisher'].unique().tolist()
    #publisher_new=[x.lower() for x in publisher]  #case sensitive
    id_num=1
    dict_new={}
    for i in publisher:
        if i not in dict_new:
            dict_new[i]=id_num
            id_num+=1
    #at this point, you will have a mapped dictionary
    #for i,v in dict_new.items():
    #    obj.writerow([i,v])
    return dict_new

def map_genre(df):
    #plug in the csv creation and header part here, publisher and corresponding ID, here you will have a csv writer object obj
    #df=pd.read_excel(CleanedData2.xlsx)
    genre=df['Genre'].unique().tolist()
    #publisher_new=[x.lower() for x in publisher]  #case sensitive
    id_num=1
    dict_new={}
    for i in genre:
        if i not in dict_new:
            dict_new[i]=id_num
            id_num+=1
    #at this point, you will have a mapped dictionary
    #for i,v in dict_new.items():
    #    obj.writerow([i,v])
    return dict_new

def map_developer(df):
    #plug in the csv creation and header part here, publisher and corresponding ID, here you will have a csv writer object obj
    #df=pd.read_excel(CleanedData2.xlsx)
    Developer=df['Developer'].unique().tolist()
    #publisher_new=[x.lower() for x in publisher]  #case sensitive
    id_num=1
    dict_new={}
    for i in Developer:
        if i not in dict_new:
            dict_new[i]=id_num
            id_num+=1
    #at this point, you will have a mapped dictionary
    #for i,v in dict_new.items():
    #    obj.writerow([i,v])
    return dict_new

def map_generic(df, column):
    #plug in the csv creation and header part here, publisher and corresponding ID, here you will have a csv writer object obj
    #df=pd.read_excel(CleanedData2.xlsx)
    Generic=df[column].unique().tolist()
    #publisher_new=[x.lower() for x in publisher]  #case sensitive
    id_num=1
    dict_new={}
    for i in Generic:
        if i not in dict_new:
            dict_new[i]=id_num
            id_num+=1
    #at this point, you will have a mapped dictionary
    #for i,v in dict_new.items():
    #    obj.writerow([i,v])
    generic_list = df[column].tolist()
    for i in range(len(generic_list)):
        generic_list[i] = dict_new[generic_list[i]]
    return generic_list

def main():
    #read_excel('CleanedData2.xlsx', "GameDataCleaned.xlsx")
    read_excel('TaggedDataUpdated.xlsx', 'NewGameDataCleaned.xlsx')

if __name__ == "__main__":
    main()