import pandas as pd
import numpy as np
#used to convert a collection of text documents to a vector of term/token counts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r"C:\STUDY\ala\dataset.csv")
print('done')
print("hi")
# DATA PROCESSING: replacing NaN values with empty string....

features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')

# combining the relevant features into a single feature.
#Next, we will define a function called combined_features. The function will combine all our useful features (keywords, cast, genres & director) from their respective rows, and return a row with all the combined features in a single string.

def combined_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

# adding new column in our original document which contain our combined features.
df["combined_features"] = df.apply(combined_features, axis =1)

print(df)












