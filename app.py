import pymysql
import pandas as pd
from preprocessing import Preprocessing
from naivebayes3 import NaiveBayes
from naivebayes3 import Vsm

db_connection = pymysql.connect(host='localhost', user='root', passwd='', database='dataset_twitter',charset='utf8')
df = pd.read_sql('SELECT * FROM tweet_2 order by id asc limit 0,9', con=db_connection)
newdf2 = pd.DataFrame()
keywordexcept = []
p = Preprocessing(keywordexcept)
for index,row in df.iterrows():
    tweet = row['text']
    pretext = p.process(tweet)
    # print("Ori : ",tweet)
    print("Preprocessed : ",pretext," -> ",row['clas'])
    df.at[index,'text'] = pretext #update column tweet in dataframe

print('\n\n\n\n\n\n\n')
# tf = TfIdf()
# tf1 = tf.tf(df,'text','clas')

# idf = tf.idf(tf1)
# nb = NaiveBayes()
# model = nb.process(idf)
# testdata = "tol bangun jalan utama sedih"
# nb.testclassification(model,testdata)

# exceptional_feature = ['clas','id']
# testdata = "tuan yg bangun nyuruh anak aja tau salam periode san"
# v = Vsm()
# vsm = v.vsm(df,exceptional_feature)
# nb = NaiveBayes()
# model = nb.builtmodel(vsm)
# nb.testclassification(model,testdata)
