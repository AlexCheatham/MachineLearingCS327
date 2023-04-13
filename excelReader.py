import pandas as pd

def read_excel():
    df = pd.read_excel('CleanedData2.xlsx')
    clean_data(df)

def clean_data(df):
    # Drop unneeded columns
    df = df.drop('Name', axis = 1)
    df = df.drop('Genre', axis = 1)
    df = df.drop('Developer', axis = 1)
    df = df.drop('Rating', axis = 1)
    
    df['Published_in_NA'] = df['NA_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_EU'] = df['EU_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_JP'] = df['JP_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df['Published_in_Other'] = df['Other_Sales'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('NA_Sales', axis = 1)
    df = df.drop('EU_Sales', axis = 1)
    df = df.drop('JP_Sales', axis = 1)
    df = df.drop('Other_Sales', axis = 1)
    # Scale user score to out of 100

    publishers = map_publisher(df)
    
    publisher_list = df['Publisher'].tolist()
    for i in range(len(publisher_list)):
        publisher_list[i] = publishers[publisher_list[i]]
    
    df['Publisher'] = publisher_list
    df['User_Score'] = df['User_Score']*10
    #read mapped file and replace publisher with their id. You may want to first convert the publisher into lower case and then 
    #write the id.
    # Saves dataframe to excel spreadsheet
    df.to_excel("GameDataCleaned.xlsx")

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

def main():
    read_excel()

if __name__ == "__main__":
    main()