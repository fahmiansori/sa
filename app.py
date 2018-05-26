import pymysql
import pandas as pd
from preprocessing import Preprocessing
from classificator import NaiveBayes
from classificator import Vsm

class App():
    def __init__(self):
        db = "dataset_twitter"
        self.training_table = "tweet_2"
        self.exceptional_feature = ['clas','id']

        self.con = pymysql.connect(host='localhost', user='root', passwd='', database=db,charset='utf8')
        self.p = Preprocessing()

    def run(self):
        df = pd.read_sql('SELECT * FROM '+self.training_table+' order by id asc limit 0,9', con=self.con)

        for index,row in df.iterrows():
            tweet = row['text']
            pretext = self.p.process(tweet)
            # print("Ori : ",tweet)
            print("Preprocessed : ",pretext," -> ",row['clas'])
            df.at[index,'text'] = pretext

        print('\n\n\n\n\n\n\n')

        testdata = "Dikejar waktu biar bisa buat Pencitraan #prihatin"
        v = Vsm()
        vsm = v.vsm(df,exceptional_feature=self.exceptional_feature)
        nb = NaiveBayes()
        model = nb.builtmodel(vsm)
        nb.classify(testdata)

    def testrun(self):
        db_connection = pymysql.connect(host='localhost', user='root', passwd='', database='dataset_twitter',charset='utf8')
        df = pd.read_sql('SELECT * FROM tweet_2 order by id asc limit 0,9', con=db_connection)
        p = Preprocessing()

        for index,row in df.iterrows():
            tweet = row['text']
            pretext = p.process(tweet)
            # print("Ori : ",tweet)
            print("Preprocessed : ",pretext," -> ",row['clas'])
            df.at[index,'text'] = pretext

        print('\n\n\n\n\n\n\n')

        exceptional_feature = ['clas','id']
        testdata = "Dikejar waktu biar bisa buat Pencitraan #prihatin"
        v = Vsm()
        vsm = v.vsm(df,exceptional_feature=exceptional_feature)
        nb = NaiveBayes()
        model = nb.builtmodel(vsm)
        nb.classify(testdata)

app = App()
app.testrun()
