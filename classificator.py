#MULTINOMIAL NB

import pandas as pd
from math import log10
from preprocessing import Preprocessing

class NaiveBayes():
    def __init__(self,con=None):
        self.model = None
        # self.preprocess = Preprocessing()

    def builtmodel(self,data,colclas='clas',qc=None):
        # data = data['vsm']
        df2 = data['vsm']
        column = data['column']
        columnlen = data['columnlen']
        totaldata = len(df2.index)
        totalclass = df2[colclas].value_counts()
        if qc:
            qc.processEvents()
        # clas = df2.clas.unique() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>perlu perbaikan, jadikan dinamis
        clas = getattr(df2,colclas)
        clas = clas.unique()
        nulval = df2[colclas].isnull().sum()
        if qc:
            qc.processEvents()

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
                        if qc:
                            qc.processEvents()
                    if qc:
                        qc.processEvents()
                totaleachwordperclas[i] = totalword_temp
                totalwordperclas[i] = cword
                if qc:
                    qc.processEvents()
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
                    if qc:
                        qc.processEvents()
                probeachwordperclas[i] = probeachwordperclas_temp
                if qc:
                    qc.processEvents()
            # print(probeachwordperclas)

            model = {'prior':prob,'cond_prob':probeachwordperclas,'clas':clas}

            self.model = model
            return model
        return False

    def classify(self,sentence,preprocess=None):
        if self.model != None:
            if preprocess is None:
                preprocess = Preprocessing()
            sentence = preprocess.process(sentence)
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

    def classifyWithModel(self,model,sentence,preprocess=None):
        if model != None:
            if preprocess is None:
                preprocess = Preprocessing()
            sentence = preprocess.process(sentence)
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

    def testclassificationDataframe(self,testdataframe,text_col,clas_col,model=None,qc=None):
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
                if qc:
                    qc.processEvents()
            confusionMatrix[c] = prediction
            tp[c] = 0
            tn[c] = 0
            fp[c] = 0
            fn[c] = 0
            if qc:
                qc.processEvents()

        for index,row in testdataframe.iterrows():
            actualclas = row[clas_col]
            testdata_token = row[text_col].split(" ")
            classify = {}
            for c in model['clas']:
                vj = model['prior'][c];
                for cc in testdata_token:
                    if cc in model['cond_prob'][c]:
                        vj*=model['cond_prob'][c][cc]
                    if qc:
                        qc.processEvents()
                classify[c] = vj
                if qc:
                    qc.processEvents()
            if qc:
                qc.processEvents()
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
                if qc:
                    qc.processEvents()

            # confusionMatrix[actualclas][argmax]+=1 # >> left = actual class, top = prediction
            confusionMatrix[argmax][actualclas]+=1 # >> left = prediction, top = actual class

            # print("Test data : ",row[text_col])
            print('Classification : ',argmax,', Actual class : ',actualclas)

        print("Confusion Matrix")
        print(confusionMatrix)

        totalclas = len(model['clas'])
        accuration = 0
        precision = 0
        recall = 0

        if totalclas > 2:

            for c in model['clas']: # > prediction kebawah (tergantung atas,opt=2)
                for cc in model['clas']: # > actual class kesamping kanan (tergantung atas,opt=2)
                    if cc == c:
                        tp[c] += confusionMatrix[c][cc]
                    else:
                        # jika axis prediction dan actual dirubah, ini juga ikut  berubah
                        fp[c] += confusionMatrix[c][cc] # acuan dari prediction axis
                        fn[cc] += confusionMatrix[c][cc] # acuan dari actual axis

                    for ccc in model['clas']:
                        if ccc != c and cc != c:
                            tn[c] += confusionMatrix[cc][ccc]
                        if qc:
                            qc.processEvents()
                    if qc:
                        qc.processEvents()
                if qc:
                    qc.processEvents()

            # for c in model['clas']:
            #     print(c,"\t",end='')
            #     for cc in model['clas']:
            #         print("\t",confusionMatrix[c][cc],end='')
            #     print("")
            #
            # for c in model['clas']:
            #     print("tp[c]",c)
            #     print(tp[c])
            #     print("fp[c]",c)
            #     print(fp[c])
            #     print("fn[c]",c)
            #     print(fn[c])
            #     print("tn[c]",c)
            #     print(tn[c])

            for c in model['clas']:
                # print("tp",c,">",tp[c],"fp",c,">",fp[c],"fn",c,">",fn[c])
                accuration += (tp[c]+tn[c])/(tp[c]+tn[c]+fp[c]+fn[c]) #1 > kurang lebih benar jika dari percobaan
                # accuration += tp[c] #2
                try:
                    precision += tp[c]/(tp[c]+fp[c])
                except:
                    precision += 0
                try:
                    recall += tp[c]/(tp[c]+fn[c])
                except:
                    recall += 0
                if qc:
                    qc.processEvents()

            print("accuration")
            print(accuration)
            print("totaldata")
            print(totaldata)
            # accuration = accuration/totaldata #2

            accuration = accuration/totalclas #1
            precision = precision/totalclas
            recall = recall/totalclas
        else:
            if totalclas == 2:
                # jika axis prediction dan actual dirubah, ini juga ikut  berubah
                tp = confusionMatrix[model['clas'][0]][model['clas'][0]]
                fn = confusionMatrix[model['clas'][1]][model['clas'][0]]
                fp = confusionMatrix[model['clas'][0]][model['clas'][1]]
                tn = confusionMatrix[model['clas'][1]][model['clas'][1]]

                accuration = (tp+tn)/(tp+tn+fp+fn)
                precision = tp/(fp+tp)
                recall = tp/(fn+tp)
            else:
                pass

        print("accuration")
        print(accuration)
        print("precision")
        print(precision)
        print("recall")
        print(recall)

        ret = {}
        ret['accuration'] = accuration
        ret['precision'] = precision
        ret['recall'] = recall

        return ret

class Vsm():
    def vsm(self,data,exceptional_feature=[],coltext='text',colclass='clas',qc=None):
        df = data
        list_feature = []
        for index,row in df.iterrows():
            text_token = row[coltext].split(" ")
            list_feature.extend(text_token)
            if qc:
                qc.processEvents()

        uniq_feature = set(list_feature)
        uniq_feature = list(uniq_feature)
        column = uniq_feature[:]
        featureDf = pd.DataFrame(columns=['feature'])
        text_t = {}
        for i in column:
            text_t['feature'] = i
            featureDf = featureDf.append(text_t,ignore_index=True)
            if qc:
                qc.processEvents()

        columnlen = len(column)
        for i in exceptional_feature:
            uniq_feature.append(i)
            if qc:
                qc.processEvents()

        df2 = pd.DataFrame(columns=uniq_feature)

        for index,row in df.iterrows():
            newdata = {}
            for i in exceptional_feature:
                newdata[i] = row[i]
                if qc:
                    qc.processEvents()

            text = row[coltext].split(" ")
            for t in text:
                for col in uniq_feature:
                    if col not in newdata:
                        newdata[col]=0
                    if t == col:
                        newdata[col]+=1
                    if qc:
                        qc.processEvents()
                if qc:
                    qc.processEvents()
            newdata[colclass] = row[colclass]
            df2 = df2.append(newdata,ignore_index=True)
            if qc:
                qc.processEvents()
        # print(df2.head())
        # df2.to_csv('file.csv') #del first column (that is index data from dataframe)

        vs_model = {'vsm':df2,'column':column,'columnlen':columnlen,'feature':featureDf}

        return vs_model
