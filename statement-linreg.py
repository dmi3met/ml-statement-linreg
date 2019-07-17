from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse import hstack


train = pd.read_csv('salary-train.csv')
y = train.iloc[:, 3]
test = pd.read_csv('salary-test-mini.csv')

train['FullDescription'] = train['FullDescription'].str.lower()
test['FullDescription'] = test['FullDescription'].str.lower()

train['FullDescription'].replace(to_replace=r'[^a-zA-Z0-9]', value=' ',
                                 inplace=True, regex=True)
test['FullDescription'].replace(to_replace=r'[^a-zA-Z0-9]', value=' ',
                                 inplace=True, regex=True)

vectorizer = TfidfVectorizer(min_df=5)
X_train_vec = vectorizer.fit_transform(train['FullDescription'])
X_test_vec = vectorizer.transform(test['FullDescription'])

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_for_train = hstack([X_train_vec, X_train_categ])
X_for_test = hstack([X_test_vec, X_test_categ])

regression = Ridge(alpha=1, random_state=241)
regression.fit(X_for_train, train['SalaryNormalized'])
predicts = regression.predict(X_for_test)
print(predicts)