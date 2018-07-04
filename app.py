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
import time

######################################
# NOTE : Definisi kata singkatan, digunakan untuk buat aturan deteksi singkatan
# Cari cara membedakan nama (pulau, tempat, dll) pada proses stemming
# Pada stemmer, terdapat kondisi stem yang mempunyai 2 removal, cari cara untuk membedakan #6,13,15,17,21,25,27,29,30
# Hilangkan karakter berlebihan, misal jaaaauuuhhh -> jauh
# ganti koneksi, sekali query hanya satu koneksi, setelah operasi selesai maka koneksi ditutup
# threshold ganti ke -1 pada seleksi fitur
# perbaiki layout tabel & tambah caption di eval serta detail waktu eksekusi eval
# optimasi for loop, cari penggantinya
# cek stopword list di preprocessing,
# [optional] implementasi multiprocessing
######################################

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

    def builtVSM(self,doFeatureSelection,take_feature,threshold,dataTraining=None,qc=None):
        if dataTraining is None:
            dataTraining = self.dataTraining

        v = Vsm()
        vsm = v.vsm(dataTraining,exceptional_feature=self.exceptional_feature,coltext=self.text_col,colclass=self.class_col,qc=qc)
        if doFeatureSelection:
            f = InfoGain()
            vsm = f.run(vsm,take_feature=take_feature,threshold=threshold,exceptional_feature=self.exceptional_feature,colclas=self.class_col,qc=qc)
        return vsm

    def preprocessing(self,doPreprocessing,doFeatureSelection,take_feature,threshold,progress,qc):
        features = None
        if self.con != None:
            if self.training_table:
                self.dataTraining = self.con.getDataAsDF(self.training_table)
                progress.setValue(10)
                if self.dataTraining is not None:
                    p = Preprocessing(con=self.con)
                    uniqFeature = []
                    features = {}
                    originalFeatureCount = 0
                    progressP = 10
                    progressS = (70-progressP)/len(self.dataTraining.index)
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
                        progressP+=progressS
                        progress.setValue(progressP)
                        # time.sleep(0.5)
                        qc.processEvents()
                    progress.setValue(70)
                    qc.processEvents()
                    uniqFeature = set(uniqFeature) # bad performance
                    qc.processEvents()
                    features['featurebefore'] = len(uniqFeature) # bad performance
                    qc.processEvents()
                    progress.setValue(80)

                    features['vsm'] = self.builtVSM(doFeatureSelection,take_feature,threshold,qc=qc)
                    progress.setValue(90)
            else:
                print("No training table!")
        progress.setValue(100)

        return features

    def trainingClassificator(self,vsm,qc=None):
        self.classificator = NaiveBayes()
        vsm = vsm['vsm']
        model = self.classificator.builtmodel(vsm,qc=qc)
        return model

    def trainingClassificatorEval(self,vsm,qc=None):
        classificator = NaiveBayes()
        classificator.builtmodel(vsm,qc=qc)
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

    def evalSentence(self,model,sentence):
        return self.classificator.classifyWithModel(model,sentence,Preprocessing(con=self.con))

    def evalKFoldCV(self,totaltrainingdata,folds,vsm=None,qc=None):
        if self.dataTraining is not None:
            dataLeft = totaltrainingdata%folds
            dataPerFold = floor(totaltrainingdata/folds)
            dataIndexFolds = []
            foldPos = 1
            dataPos = 1
            if qc:
                qc.processEvents()
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
                if qc:
                    qc.processEvents()

            if dataLeft > 0:
                indexToAdd = 0
                for i in range((totaltrainingdata-dataLeft),totaltrainingdata):
                    dataIndexFolds[indexToAdd].append(i)
                    indexToAdd+=1
                    if qc:
                        qc.processEvents()

            evalResults = []
            for i in range(folds):
                dataIndexFolds_copy = list(dataIndexFolds)
                testData = self.dataTraining.iloc[dataIndexFolds_copy[i]]
                del dataIndexFolds_copy[i]
                trainingDataList = []
                for j in dataIndexFolds_copy:
                    trainingDataList.extend(j)
                    if qc:
                        qc.processEvents()
                trainingData = self.dataTraining.iloc[trainingDataList]
                if vsm:
                    vsmTraining = vsm
                else:
                    vsmTraining = self.builtVSM(False,0,0,dataTraining=trainingData)
                model = self.trainingClassificatorEval(vsmTraining,qc=qc)
                evalResult = model.testclassificationDataframe(testData,self.text_col,self.class_col,qc=qc)
                evalResults.append(evalResult)
                if qc:
                    qc.processEvents()

            avg_accuration = 0
            avg_precision = 0
            avg_recall = 0
            for i in evalResults:
                avg_accuration += i['accuration']
                avg_precision += i['precision']
                avg_recall += i['recall']
                if qc:
                    qc.processEvents()
            if len(evalResults) > 0:
                er = len(evalResults)
                avg_accuration = avg_accuration/er
                avg_precision = avg_precision/er
                avg_recall = avg_recall/er
            ret = {}
            ret['accuration'] = float(format(avg_accuration,'.2f'))*100
            ret['precision'] = float(format(avg_precision,'.2f'))
            ret['recall'] = float(format(avg_recall,'.2f'))

            # print(dataIndexFolds)
            return ret

        else:
            print("No data training found!")
        return False
