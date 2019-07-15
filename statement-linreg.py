from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

'''
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)



Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.
Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. Целевая переменная записана в столбце SalaryNormalized.

4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.

Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.42. При необходимости округляйте дробную часть до двух знаков.



'''

# Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из
# файла salary-train.csv (либо его заархивированную версию salary-train.zip).
data = pd.read_csv('salary-train.csv')
y = data.iloc[:, 3]
X = data.iloc[:, 0:2]

# Проведите предобработку:
# Приведите тексты к нижнему регистру (text.lower()).
data['FullDescription'] = data['FullDescription'].str.lower()

# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее
# разделение текста на слова. Для такой замены в строке text подходит
# следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text).
# Также можно воспользоваться методом replace у DataFrame,
# чтобы сразу преобразовать все тексты:
data['FullDescription'].replace(to_replace=r'[^a-zA-Z0-9]', value=' ',
                                inplace=True, regex=True)
print(data['FullDescription'])
# Примените TfidfVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы
# в 5 объектах (параметр min_df у TfidfVectorizer).

