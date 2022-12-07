
















import pandas as pd
import numpy as np
import re
#used to convert a collection of text documents to a vector of term/token counts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_csv(r"C:\STUDY\ala\dataset.csv")
print('done')

# DATA PROCESSING: replacing NaN values with empty string....

features = ['cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')

# combining the relevant features into a single feature.
#Next, we will define a function called combined_features. The function will combine all our useful features (keywords, cast, genres & director) from their respective rows, and return a row with all the combined features in a single string.

def combined_features(row):
    return row['cast']+" "+row['genres']+" "+row['director']

# adding new column in our original document which contain our combined features.
df["combined_features"] = df.apply(combined_features, axis =1)

#extracting features and converting it into the language that is supported by machine learning. we convert the textual data into the matrices based on the repitition of texts.
tokenized_documents = [re.findall(r'\w+', d.lower()) for d in df["combined_features"]]

# cv = CountVectorizer()
# count_matrix = cv.fit_transform(df["combined_features"])
# print('count matrix:', count_matrix.toarray())

# print(give_length())
# cosine_sim = cosine_similar(count_matrix)



lexicon = sorted(set(sum(tokenized_documents, [])))

# our lexicon or vocabulary looks like this.


########


from collections import OrderedDict

vector_template = OrderedDict((token, 0) for token in lexicon)

# our vector template looks like this.


#######

import copy
from collections import Counter

doc_tfidf_vectors = []
for doc_tokens in tokenized_documents:
    vec = copy.copy(vector_template)
    token_counts = Counter(doc_tokens)
    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc_tokens in tokenized_documents:
            if key in _doc_tokens:
                docs_containing_key += 1
        tf = value / len(doc_tokens)
        if docs_containing_key:
            idf = len(tokenized_documents) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    doc_tfidf_vectors.append(vec)
    
# and our document vectors are the following


########




import math

def cosine_sim(vec1, vec2):
    vec1 = list(vec1.values())
    vec2 = list(vec2.values())
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    if mag_1 == 0:
        mag_1 = 1
    if mag_2 == 0:
        mag_2 = 1
    return dot_prod / (mag_1 * mag_2)

# Lets compare the first document to the other two. As expected
# document 1 and 2 have some similarity, documents 1 and 3 have
# 0 similarity since they don't share any words.







for r, doc1 in enumerate(doc_tfidf_vectors):
    print(r, end='\t')
    for c, doc2 in enumerate(doc_tfidf_vectors):
        print(round(cosine_sim(doc1, doc2), 2), end='\t')
    print()











