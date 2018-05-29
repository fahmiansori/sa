#MULTINOMIAL NB

import pandas as pd
from math import log10
from preprocessing import Preprocessing

class NaiveBayes():
    def __init__(self):
        self.model = None
        self.preprocess = Preprocessing()

    def builtmodel(self,data,colclas='clas'):
        # data = data['vsm']
        df2 = data['vsm']
        column = data['column']
        columnlen = data['columnlen']
        totaldata = len(df2.index)
        totalclass = df2[colclas].value_counts()
        # clas = df2.clas.unique() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>perlu perbaikan, jadikan dinamis
        clas = getattr(df2,colclas)
        clas = clas.unique()
        nulval = df2[colclas].isnull().sum()

        if totalclass.empty or nulval > 0:
            print("No class found!")
        else:
            prob = {}
            totalwordperclas = {}
            totaleachwordperclas = {}

            for i in clas:
                prob[i] = totalclass[i]/totaldata
                dfclas = df2.loc[df2[colclas] == i]
                cword = 0
                cword2 = 0
                totalword_temp = {}
                for index,row in dfclas.iterrows():
                    for cc in column:
                        cword+=int(row[cc])

                        cword2+=int(row[cc])
                        if cc in totalword_temp:
                            totalword_temp[cc] += cword2
                        else:
                            totalword_temp[cc] = cword2
                        cword2 = 0
                totaleachwordperclas[i] = totalword_temp
                totalwordperclas[i] = cword
            # print(prob)
            # print(totalwordperclas)
            # print(totaleachwordperclas)

            probeachwordperclas = {}
            #likelihood
            for i in clas:
                p = 0
                probeachwordperclas_temp = {}
                for cc in column:
                    p = (totaleachwordperclas[i][cc]+1)/(totalwordperclas[i]+columnlen)
                    probeachwordperclas_temp[cc] = p
                probeachwordperclas[i] = probeachwordperclas_temp
            # print(probeachwordperclas)

            model = {'prior':prob,'cond_prob':probeachwordperclas,'clas':clas}

            self.model = model
            return model
        return False

    def classify(self,sentence):
        if self.model != None:
            sentence = self.preprocess.process(sentence)
            sentence_split = sentence.split(" ")

            clas = {}
            for c in self.model['clas']:
                vj = self.model['prior'][c];
                for cc in sentence_split:
                    if cc in self.model['cond_prob'][c]:
                        vj*=self.model['cond_prob'][c][cc]
                clas[c] = vj

            i = 0
            prev = 0;
            curr = 0;
            argmax = ''
            for c in self.model['clas']:
                curr = clas[c];
                if(curr > prev):
                    argmax = c
                    prev = curr

            print("Test data : ",sentence)
            print('Class : ',argmax)

            return argmax

        else:
            print("No model!")

        return False

    def classifyWithModel(self,model,sentence):
        if model != None:
            sentence = self.preprocess.process(sentence)
            sentence_split = sentence.split(" ")

            clas = {}
            for c in model['clas']:
                vj = model['prior'][c];
                for cc in sentence_split:
                    if cc in model['cond_prob'][c]:
                        vj*=model['cond_prob'][c][cc]
                clas[c] = vj

            i = 0
            prev = 0;
            curr = 0;
            argmax = ''
            for c in model['clas']:
                curr = clas[c];
                if(curr > prev):
                    argmax = c
                    prev = curr

            print("Test data : ",sentence)
            print('Class : ',argmax)

            return argmax

        else:
            print("No model!")

        return False

    def testclassification(self,model,testdata,actualclas=''):
        testdata_token = testdata.split(" ")
        classify = {}
        for c in model['clas']:
            vj = model['prior'][c];
            for cc in testdata_token:
                if cc in model['cond_prob'][c]:
                    vj*=model['cond_prob'][c][cc]
            classify[c] = vj
        # print(classify)

        i = 0
        prev = 0;
        curr = 0;
        argmax = ''
        for c in model['clas']:
            curr = classify[c];
            if(curr > prev):
                argmax = c
                prev = curr

        print("Test data : ",testdata)
        print('Classification : ',argmax)

        if actualclas:
            if actualclas == argmax:
                return True

        return False

    def testclassificationDataframe(self,testdataframe,text_col,clas_col,model=None):
        if model is None:
            model = self.model

        totaldata = len(testdataframe.index)
        confusionMatrix = {}
        tp = {}
        tn = {}
        fp = {}
        fn = {}
        for c in model['clas']:
            prediction = {}
            for cc in model['clas']:
                prediction[cc] = 0
            confusionMatrix[c] = prediction
            tp[c] = 0
            tn[c] = 0
            fp[c] = 0
            fn[c] = 0

        for index,row in testdataframe.iterrows():
            actualclas = row[clas_col]
            testdata_token = row[text_col].split(" ")
            classify = {}
            for c in model['clas']:
                vj = model['prior'][c];
                for cc in testdata_token:
                    if cc in model['cond_prob'][c]:
                        vj*=model['cond_prob'][c][cc]
                classify[c] = vj
            # print(classify)

            i = 0
            prev = 0;
            curr = 0;
            argmax = ''
            for c in model['clas']:
                curr = classify[c];
                if(curr > prev):
                    argmax = c
                prev = curr

            confusionMatrix[actualclas][argmax]+=1

            # print("Test data : ",row[text_col])
            # print('Classification : ',argmax,', Actual class : ',actualclas)

        # print("Confusion Matrix")
        # print(confusionMatrix)
        for c in model['clas']: # > actual class kebawah
            for cc in model['clas']: # > prediction kesamping kanan
                if cc == c:
                    tp[c] += confusionMatrix[c][cc]
                else:
                    fp[cc] += confusionMatrix[c][cc]
                    fn[c] += confusionMatrix[c][cc]

                for ccc in model['clas']:
                    if ccc != c and cc != c:
                        tn[c] += confusionMatrix[cc][ccc]

        totalclas = len(model['clas'])
        accuration = 0
        precision = 0
        avg_precision = 0
        recall = 0
        avg_recall = 0
        for c in model['clas']:
            # print("tp",c,">",tp[c],"fp",c,">",fp[c],"fn",c,">",fn[c])
            accuration += tp[c]
            try:
                precision = tp[c]/(tp[c]+fp[c])
            except:
                precision = 0
            avg_precision += precision
            try:
                recall = tp[c]/(tp[c]+fn[c])
            except:
                recall = 0
            avg_recall += recall
        accuration = accuration/totaldata
        avg_precision = avg_precision/totalclas
        avg_recall = avg_recall/totalclas
        ret = {}
        ret['accuration'] = accuration
        ret['precision'] = avg_precision
        ret['recall'] = avg_recall

        return ret

class Vsm():
    def vsm(self,data,exceptional_feature=[],coltext='text',colclass='clas'):
        df = data
        list_feature = []
        for index,row in df.iterrows():
            text_token = row[coltext].split(" ")
            list_feature.extend(text_token)

        uniq_feature = set(list_feature)
        uniq_feature = list(uniq_feature)
        column = uniq_feature[:]
        featureDf = pd.DataFrame(columns=['feature'])
        text_t = {}
        for i in column:
            text_t['feature'] = i
            featureDf = featureDf.append(text_t,ignore_index=True)

        columnlen = len(column)
        for i in exceptional_feature:
            uniq_feature.append(i)

        df2 = pd.DataFrame(columns=uniq_feature)

        for index,row in df.iterrows():
            newdata = {}
            for i in exceptional_feature:
                newdata[i] = row[i]

            text = row[coltext].split(" ")
            for t in text:
                for col in uniq_feature:
                    if col not in newdata:
                        newdata[col]=0
                    if t == col:
                        newdata[col]+=1
            newdata[colclass] = row[colclass]
            df2 = df2.append(newdata,ignore_index=True)
        # print(df2.head())
        # df2.to_csv('file.csv') #del first column (that is index data from dataframe)

        vs_model = {'vsm':df2,'column':column,'columnlen':columnlen,'feature':featureDf}

        return vs_model
