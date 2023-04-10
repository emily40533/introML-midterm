import joblib
model_pretrained = joblib.load('house-20230410.pkl')
import pandas as pd
df_test = pd.read_csv("data/test.csv")
cols_to_keep = ['Id','1stFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GrLivArea', 'GarageArea', 'GarageCars']
df_test.drop(columns=[col for col in df_test.columns if col not in cols_to_keep], inplace=True)
df_test.info()


df_test.loc[:, 'TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].median())
df_test.loc[:, 'GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].median())
df_test.loc[:, 'GarageArea'] = df_test['GarageArea'].fillna(df_test['GarageArea'].median())

prediction2 = model_pretrained.predict(df_test)
prediction2

forSubmissionDF = pd.DataFrame(columns=['Id', 'SalePrice'])
forSubmissionDF 
start_id = df_test['Id'].min()
end_id = df_test['Id'].max()
print(start_id, end_id)
forSubmissionDF['Id'] = range(start_id, end_id+1)
forSubmissionDF['SalePrice']=prediction2

forSubmissionDF.to_csv('HouseSubmission_20230410.csv', index=False)