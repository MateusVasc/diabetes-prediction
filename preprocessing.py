import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

class Preprocessing:
    def __init__(self, df):
        self.df = df

    def combine_race_columns(self, prefix="race:"):
        race_columns = [col for col in self.df.columns if col.startswith(prefix)]
        self.df['race'] = self.df[race_columns].idxmax(axis=1).apply(lambda x: x.split(':')[1])
        self.df.drop(columns=race_columns, inplace=True)
        return self.df
    
    def one_hot_encode_features(self, columns):
        ohe = OneHotEncoder(sparse_output=False)

        for col in columns:
            if col in self.df.columns:
                encoded = ohe.fit_transform(self.df[[col]])
                
                col_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                
                encoded_df = pd.DataFrame(encoded, columns=col_names, index=self.df.index)
                
                self.df = pd.concat([self.df, encoded_df], axis=1)
                
                self.df.drop(columns=[col], inplace=True)
            else:
                print(f"Coluna {col} não encontrada no DataFrame.")

        return self.df

    def target_encode_features(self, feature_cols, target_col):
        te = TargetEncoder()
        
        for col in feature_cols:
            if col in self.df.columns:
                self.df[col] = te.fit_transform(self.df[col], self.df[target_col])
            else:
                print(f"Coluna {col} não encontrada no DataFrame.")
        
        return self.df
