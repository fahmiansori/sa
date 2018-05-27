import pymysql
import pandas as pd
from preprocessing import Preprocessing
from classificator import NaiveBayes
from classificator import Vsm
from feature_selection import InfoGain
from db import Database

class App():
    def __init__(self):
        # db = "dataset_twitter"
        # self.training_table = "tweet_2"
        # self.exceptional_feature = ['clas','id']
        # self.class_col = 'clas'
        # self.text_col = 'text'
        #
        # self.con = pymysql.connect(host='localhost', user='root', passwd='', database=db,charset='utf8')
        # # check point > tambah exception handling waktu connect db dan query
        # self.p = Preprocessing()
        # self.feature_selection = InfoGain()
        # cursor.execute("select ffrom aldas where dsalkd='{0}'".format(iniwhere))

        self.db = ""
        self.training_table = ""
        self.exceptional_feature = []
        self.class_col = 'clas'
        self.text_col = 'text'

        self.con = None
        self.preprocessing = None
        self.feature_selection = None

    def checkConnection(self):
        if self.con != None:
            return True
        return False

    def connectDb(self,host,user,password,db):
        tables = None
        tryConnect = Database()
        tryConnectStat = tryConnect.connect(host,user,password,db)
        if tryConnectStat['success'] == True:
            self.con = tryConnect
            tables = self.con.tables(db)
        else:
            self.con = None

        ret = {}

        ret['success'] = tryConnectStat['success']
        ret['msg'] = tryConnectStat['msg']
        ret['tables'] = tables
        return ret

    def run(self):
        df = pd.read_sql('SELECT * FROM '+self.training_table+' order by id asc limit 0,9', con=self.con)
        # df = pd.read_sql('SELECT * FROM '+self.training_table+' order by id asc', con=self.con) # activate if deploying

        for index,row in df.iterrows():
            tweet = row[self.text_col]
            pretext = self.p.process(tweet)
            # print("Ori : ",tweet)
            print("Preprocessed : ",pretext," -> ",row[self.class_col])
            df.at[index,'text'] = pretext

        print('\n\n\n\n\n\n\n')

        v = Vsm()
        vsm = v.vsm(df,exceptional_feature=self.exceptional_feature)
        vsm = self.feature_selection.run(vsm,take_feature=10,exceptional_feature=self.exceptional_feature)
        nb = NaiveBayes()
        model = nb.builtmodel(vsm)

        testdata = "kecewa tolong hati jalan kalimant tidak rubah musim hujan parah"
        nb.classify(testdata)

    def run2(self):
        df = pd.read_sql('SELECT * FROM '+self.training_table+' order by id asc limit 0,9', con=self.con)

        for index,row in df.iterrows():
            tweet = row['text']
            pretext = self.p.process(tweet)
            # print("Ori : ",tweet)
            # print("Preprocessed : ",pretext," -> ",row['clas'])
            df.at[index,'text'] = pretext

        print('\n\n\n\n\n\n\n')

        # testdata = "Dikejar waktu biar bisa buat Pencitraan #prihatin"
        v = Vsm()
        vsm = v.vsm(df,exceptional_feature=self.exceptional_feature)
        vsm = self.feature_selection.run(vsm,exceptional_feature=self.exceptional_feature)

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

# app = App()
# app.run()
