
















import pandas as pd
import numpy as np
#used to convert a collection of text documents to a vector of term/token counts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def equal_size(l1,l2):
    ms = max(l1.shape[0],l2.shape[0])


def dot_product(l1,l2):
    ans = []
    assert l1.shape[0] == l2.shape[0]
    for i in range(l1.shape[0]):
        if l1[i] == 0 or l2[i] == 0:
            pass
        else:
            ans.append(l1[i]*l2[i])
    return sum(ans)
def magnitude(l1):
    ans = 0
    for i in range(len(l1)):
        ans += (l1[i])**2
    return ans**(0.5)

def cosine_similar(count_matrix):
    length = count_matrix.shape[0]
    cosine_sim = [[None]*4815]*4815
    for i in range(4815):
        for j in range(4815):
            dot = dot_product(count_matrix[i],count_matrix[j])
            l1 = count_matrix[i]
            l2 = count_matrix[j]
            cosine_sim[i][j] = dot/(magnitude(l1)*magnitude(l2))
    return cosine_sim

def give_length(arr):
    ans = []
    for i in range(arr.shape[0]):
        ans.append(arr[i].shape[1])
    return ans





df = pd.read_csv(r"C:\STUDY\ala\dataset.csv")
print('done')

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

#extracting features and converting it into the language that is supported by machine learning. we convert the textual data into the matrices based on the repitition of texts.
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
print('count matrix:', count_matrix.toarray())
print('--------')
print(give_length(count_matrix))

cosine_sim = cosine_similar(count_matrix)























