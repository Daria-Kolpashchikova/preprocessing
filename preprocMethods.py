import re
import nltk
from gensim.models import Word2Vec
from zipfile import ZipFile
import pandas as pd

def calculateBOW(wordset, l_doc):
  tf_diz = dict.fromkeys(wordset,0)
  for word in l_doc:
      tf_diz[word]=l_doc.count(word)
  return tf_diz

#чтение данных из архива
archive = ZipFile('bbc-fulltext.zip', 'r')
fileNames = archive.namelist()
fileTexts = [archive.read(name) for name in archive.namelist()]

#преобразование данных к нужному виду
string = ' '.join([str(item) for item in fileTexts])
processed_article = string.lower()

processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)

#подготовка параметров для векторизации
all_sentences = nltk.sent_tokenize(processed_article)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

#применение библиотеки w2v
w2v_model = Word2Vec(all_words, min_count=2)

#проверка - найти слова, похожие на time
sim_words = w2v_model.wv.most_similar('time')
print(sim_words)

#модель bad of words
bow = calculateBOW(string, fileTexts)
df_bow = pd.DataFrame([bow])
print(df_bow)
