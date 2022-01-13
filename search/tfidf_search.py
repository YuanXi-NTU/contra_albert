import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# tf-idf search


path='./data/'
stopwords=open(os.getcwd()+'/stop.txt','r',encoding='utf-8').read().split('\n')
# vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=stopwords)
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',
                             stop_words=stopwords,
                             min_df=2)
tf_idf_transformer = TfidfTransformer()
lines=[]
for i in os.listdir(path):
    f=open(path+i,'r',encoding='utf-8').read().split('\n')
    for i in f:
        lines.append(i)

X = vectorizer.fit_transform(lines)
X_sparse = scipy.sparse.csr_matrix(X)

# print(len(vectorizer.get_feature_names()))

sample="明星 回应 爱国 问题 标准 来"
check_stop=set(stopwords)
def search(query):
    query_vec=vectorizer.transform([query])
    cos_sim = cosine_similarity(X_sparse,query_vec).reshape(-1)
    res = np.argsort(cos_sim)[-1:-11:-1]
    ans=[{"score": cos_sim[i], "text": lines[i]} for i in res]
    if abs(ans[0]['score']-0)<1e-6:
        ans[0]={"score": 0, "text": 'query OOV'}
        # ans[1]={"score": 0, "text": 'query OOV'}
    return ans

search(sample)

