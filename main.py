import pandas as pd
import sklearn.model_selection as sm
from keras.layers import Flatten
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import tensorflow as tf
# from IPhyton.display import display
from TokenizateUA import TokenizateUA

#Все parquet файлы переехали на гугл диск:
#https://drive.google.com/drive/folders/18FbIpfbq0cRKPw5a3cBdjFVTXkdHUiut?usp=sharing
table = pd.read_parquet('../parquets/train.parquet')

#разделение выборки на тестовую и обучающую
x_train, x_test, y_train, y_test=sm.train_test_split(table, table['label'], test_size = 0.3, random_state=1)

#Создание отдельного списка юзер-агентов(пустые юзер-агенты будут добавляться как пустые строки)
ualist=[]
for i in range(len(x_train['ua'])):
    toks=[]
    try:
        #разбиваем юзер-агент на токены и добавляем их в ualist
        for j in TokenizateUA(x_train['ua'][i]):
            toks.append(j)
        ualist.append(toks)
    except KeyError:
        ualist.append([''])

print('Preparing data and tokenizing.\n')

#culist - список наборов кривых из обучающей выборки
culist = x_train['curves'].astype(str).tolist()
culist = [elem.replace('\'', '') for elem in culist]
culist = [elem.replace('[', '') for elem in culist]
culist = [elem.replace(']', '') for elem in culist]
culist = [elem.split() for elem in culist]

#culistt - список наборов кривых из тестовой выборки
culistt = x_test['curves'].astype(str).tolist()
culistt = [elem.replace('\'', '') for elem in culistt]
culistt = [elem.replace('[', '') for elem in culistt]
culistt = [elem.replace(']', '') for elem in culistt]
culistt = [elem.split() for elem in culistt]

#Добавление префикса к кривым, поскольку названия некоторых кривых пересекаются с названиями шифров
#А это даёт неправильное количество уникальных шифров+кривых
for i in range(len(culist)):
    for j in range(len(culist[i])):
        culist[i][j]='curve:'+culist[i][j]

for i in range(len(culistt)):
    for j in range(len(culistt[i])):
        culistt[i][j]='curve:'+culistt[i][j]

#Аналогичные списки для наборов шифров из обучающей и тестовой выборок
clist = x_train['ciphers'].astype(str).tolist()
clist = [elem.replace('\'', '') for elem in clist]
clist = [elem.replace('[', '') for elem in clist]
clist = [elem.replace(']', '') for elem in clist]
clist = [elem.split() for elem in clist]

clistt = x_test['ciphers'].astype(str).tolist()
clistt = [elem.replace('\'', '') for elem in clistt]
clistt = [elem.replace('[', '') for elem in clistt]
clistt = [elem.replace(']', '') for elem in clistt]
clistt = [elem.split() for elem in clistt]

#склеивание в один набор: user-agent + ciphers + curves
for i in range(len(clist)):
    clist[i]=ualist[i] + clist[i] +culist[i]

for i in range(len(clistt)):
    clistt[i]=ualist[i] + clistt[i] +culistt[i]

#токенизация склеинных наборов
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clist)
word_index = tokenizer.word_index
total_unique_words = len(tokenizer.word_index) + 1
vocab = list(tokenizer.word_index.keys())

#Приведение к прямоугольному виду для преобразования в тензор
mlen = 0
for i in clist:
    mlen = max(mlen, len(i))
for j in range(len(clist)):
    if len(clist[j]) < mlen:
        dif = mlen - len(clist[j])
        for _ in range(dif):
            clist[j].append('')

for i in clistt:
    for j in range(len(clistt)):
        if len(clistt[j]) < mlen:
            dif = mlen - len(clistt[j])
            for _ in range(dif):
                clistt[j].append('')

#Получившийся тензор, с которым можно работать
data = tf.constant(clist)
data_test=tf.constant(clistt)

print('Preparing model\n')
model=Sequential()

#input_l - входной слой
#strtovec - слой отображающий строковые признаки в числовые
#embedding - слой, создающий эмбеддинг из числовых признаков, полученных из strtovec
#lstm - maximize the perfomance
#predictions - результат предсказания
input_l=tf.keras.Input(shape=(len(clist[0]),))
strtovec = tf.keras.layers.StringLookup(vocabulary=vocab)
embedding = tf.keras.layers.Embedding(input_dim=total_unique_words, output_dim=4, input_length=len(clist[0])) #output dim 4
lstm = tf.keras.layers.Bidirectional(LSTM(4, return_sequences=True, dropout=0.2)) #input_shape=(33562, 102, 4)
predictions = Dense(1, activation='sigmoid')

#Добавление слоёв
model.add(input_l)
model.add(embedding)
model.add(Dense(4, activation='relu'))
model.add(lstm)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(predictions)

#компиляция
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #rmsprop

#Обучение модели.
#Максимальный результат 0.681 на test.parquet был получен на 25 эпохах, batch 128 и val split 0.1
history = model.fit(strtovec(tf.constant(clist)), y_train, epochs=15, batch_size=256, validation_split=0.2) #5 32 0.1
#10 16 0.2  - 0.8687
#10 256 0.1 - 0.8662
#25 128 0.1 - 0.8833
#25 128 0.1 - 0.8757
#25 128 0.1 - 0.8808
#30 128 0.1 - 0.8872 subs
#30 128 0.1 - 0.8953 subs2
#30 128 0.1 - 0.9002 subs3

print('Now testing.\n')
pred=model.predict(strtovec(data_test))
score = model.evaluate(strtovec(data_test), y_test, batch_size=64)
print(score)

test=pd.read_parquet('../parquets/test.parquet')

#Загрузка и преобразование тестовых данных из test.parquet
tstculist=[]
for i in range(len(test['curves'])):
    tstculist.append(test['curves'][i].decode(encoding='utf-8').replace(']', '').replace('[', '').replace(',', ' ').replace('\"', '').split(' '))
for i in range(len(tstculist)):
    for j in range(len(tstculist[i])):
        tstculist[i][j]="curve:" + tstculist[i][j]

tstualist=[]
for i in range(len(test['ua'])):
    toks=[]
    try:
        for j in TokenizateUA(test['ua'][i]):
            toks.append(j)
        tstualist.append(toks)
    except KeyError:
        tstualist.append([''])

tstclist=[]
for i in range(len(test['ciphers'])):
    tstclist.append(test['ciphers'][i].decode(encoding='utf-8').replace(']', '').replace('[', '').replace(',', ' ').replace('\"', '').split(' '))
for h in range(len(tstclist)):
    tstclist[h]=tstualist[h] + tstclist[h] +tstculist[h]

mlen = 0
for i in tstclist:
    mlen = max(mlen, len(i))
mlen-=1
print("mlen: "+ str(mlen))
for j in range(len(tstclist)):
    if len(tstclist[j]) <= mlen:
        dif = mlen - len(tstclist[j])
        for _ in range(dif):
            tstclist[j].append('')
    else:
        print("yes\n")
        diff=len(tstclist[j])-mlen
        tstclist[j]=tstclist[j][:-diff:]
        print(len(tstclist[j]))

tstdata=tf.constant(tstclist)

tstpred=model.predict(strtovec(tstdata))

print('writing results.\n')

#Запись результатов в csv файл
ans = {"id": [], "is_bot": []}
for i in range(len(test)):
    ans['id'].append(test['id'][i])
    ans['is_bot'].append(tstpred[i][0])
res = pd.DataFrame(ans)
res.to_csv("submission.csv", index=False)