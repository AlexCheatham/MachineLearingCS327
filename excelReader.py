import pandas as pd

def read_excel():
    df = pd.read_excel('CleanedData2.xlsx')
    clean_data(df)

def clean_data(df):
    # Drop unneeded columns
    df = df.drop('Name', axis = 1)
    df = df.drop('Genre', axis = 1)
    df = df.drop('Publisher', axis = 1)
    df = df.drop('Developer', axis = 1)
    df = df.drop('Rating', axis = 1)
    
    # Scale user score to out of 100
    df['User_Score'] = df['User_Score']*10
    
    # Saves dataframe to excel spreadsheet
    df.to_excel("GameDataCleaned.xlsx")

def main():
    read_excel()

if __name__ == "__main__":
    main()