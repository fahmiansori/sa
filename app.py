# ░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓░░░░
# ░░░░░░░░░░░░░░░░░░░░░░░▓▓░▓▓▓░▓░░
# ░░░░░░░░░░░░░░░░▓░▓░░▓▓▓░▓▓▓░░▓▓░
# ░░░░░░░░░░░▓▓▓▓▓▓░▓░▓▓▓░░▓▓▓░▓▓▓▓
# ░░░░░░▓▓▓▓▓▓▓▓▓░░▓▓░▓▓▓░░▓▓▓░▓░▓▓
# ░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░▓▓▓▓░▓▓▓▓░░░▓▓
# ░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓▓░▓▓▓▓▓░░░▓▓
# ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓░░▓▓▓▓░░░▓▓▓
# ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓▓░░░▓▓▓░
# ░░▓▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░▓▓▓░░
# ░░▓▓░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓▓▓░░░
# ░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░
# ░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░
# ░░░░░░░░░░░░░░░▓▓▓▓▓░░░░░░░░░░░░░
# ░░░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░
# ░░░▓▓░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░
# ░░▓▓▓░▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░
# ░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░
# ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░﻿

import pymysql
import pandas as pd
from preprocessing import Preprocessing
from classificator import NaiveBayes
from classificator import Vsm
from feature_selection import InfoGain
from db import Database
from math import floor

class App():
    def __init__(self):
        self.db = ""
        self.training_table = ""
        self.exceptional_feature = []
        self.class_col = 'clas'
        self.text_col = 'text'

        self.con = None
        self.classificator = None
        self.dataTraining = None

    def checkConnection(self):
        if self.con != None:
            return True
        return False

    def connectTo(self,host,user,password,db): # connect to different db and return the connection
        tryConnect = Database()
        tryConnectStat = tryConnect.connect(host,user,password,db)
        if tryConnectStat['success'] == True:
            return tryConnect

        return None

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

    def setTrainingTable(self,table):
        self.training_table = table
    def setExceptionalFeature(self,ex):
        self.exceptional_feature = ex
    def setClassCol(self,col):
        self.class_col = col
    def setTextCol(self,col):
        self.text_col = col

    def builtVSM(self,doFeatureSelection,take_feature,threshold,dataTraining=None):
        if dataTraining is None:
            dataTraining = self.dataTraining

        v = Vsm()
        vsm = v.vsm(dataTraining,exceptional_feature=self.exceptional_feature,coltext=self.text_col,colclass=self.class_col)
        if doFeatureSelection:
            f = InfoGain()
            vsm = f.run(vsm,take_feature=take_feature,threshold=threshold,exceptional_feature=self.exceptional_feature,colclas=self.class_col)
        return vsm

    def preprocessing(self,doPreprocessing,doFeatureSelection,take_feature,threshold):
        features = None
        if self.con != None:
            if self.training_table:
                self.dataTraining = self.con.getDataAsDF(self.training_table)
                if self.dataTraining is not None:
                    p = Preprocessing()
                    uniqFeature = []
                    features = {}
                    originalFeatureCount = 0
                    for index,row in self.dataTraining.iterrows():
                        text = row[self.text_col]
                        t = p.processNoPre(text).split(" ") # bad performance
                        uniqFeature.extend(t) # bad performance

                        if doPreprocessing:
                            pretext = p.process(text)
                        else:
                            pretext = p.processNoPre(text)
                        # print("Ori : ",text)
                        # print("Preprocessed : ",pretext," -> ",row[self.class_col])
                        self.dataTraining.at[index,self.text_col] = pretext
                    uniqFeature = set(uniqFeature) # bad performance
                    features['featurebefore'] = len(uniqFeature) # bad performance

                    features['vsm'] = self.builtVSM(doFeatureSelection,take_feature,threshold)

            else:
                print("No training table!")

        return features

    def trainingClassificator(self,vsm):
        self.classificator = NaiveBayes()
        vsm = vsm['vsm']
        model = self.classificator.builtmodel(vsm)
        return model

    def trainingClassificatorEval(self,vsm):
        classificator = NaiveBayes()
        classificator.builtmodel(vsm)
        return classificator

    def getDataTrainingProperty(self,clas):
        ret = {}
        if self.dataTraining is not None:
            ret['totaltrainingdata'] = len(self.dataTraining.index)
            t = self.dataTraining[self.class_col].value_counts()
            tdpc = ""
            num = 0
            tlen = len(t)
            for c in clas:
                tdpc+=c+" : "+str(t[c])
                num+=1
                if num < tlen:
                    tdpc+=", "
            ret['totaltrainingdataperclas'] = tdpc
            return ret
        return False

    def evalKFoldCV(self,totaltrainingdata,folds):
        if self.dataTraining is not None:
            dataLeft = totaltrainingdata%folds
            dataPerFold = floor(totaltrainingdata/folds)
            dataIndexFolds = []
            foldPos = 1
            dataPos = 1
            for i in range(totaltrainingdata):
                if dataPos == 1:
                    newFold = []
                if dataPos <= dataPerFold:
                    newFold.append(i)
                    dataPos+=1
                    if dataPos > dataPerFold:
                        dataPos=1
                if dataPos == 1:
                    dataIndexFolds.append(newFold)
                    foldPos+=1
                if foldPos > folds:
                    break

            if dataLeft > 0:
                indexToAdd = 0
                for i in range((totaltrainingdata-dataLeft),totaltrainingdata):
                    dataIndexFolds[indexToAdd].append(i)
                    indexToAdd+=1

            evalResults = []
            for i in range(folds):
                dataIndexFolds_copy = list(dataIndexFolds)
                testData = self.dataTraining.iloc[dataIndexFolds_copy[i]]
                del dataIndexFolds_copy[i]
                trainingDataList = []
                for j in dataIndexFolds_copy:
                    trainingDataList.extend(j)
                trainingData = self.dataTraining.iloc[trainingDataList]
                vsmTraining = self.builtVSM(False,0,0,dataTraining=trainingData)
                model = self.trainingClassificatorEval(vsmTraining)
                evalResult = model.testclassificationDataframe(testData,self.text_col,self.class_col)
                evalResults.append(evalResult)

            avg_accuration = 0
            avg_precision = 0
            avg_recall = 0
            for i in evalResults:
                avg_accuration += i['accuration']
                avg_precision += i['precision']
                avg_recall += i['recall']
            er = len(evalResults)
            avg_accuration = avg_accuration/er
            avg_precision = avg_precision/er
            avg_recall = avg_recall/er
            ret = {}
            ret['accuration'] = float(format(avg_accuration,'.2f'))
            ret['precision'] = float(format(avg_precision,'.2f'))
            ret['recall'] = float(format(avg_recall,'.2f'))

            # print(dataIndexFolds)
            return ret

        else:
            print("No data training found!")
        return False
