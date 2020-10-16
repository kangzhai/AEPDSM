# Adaptive selective ensemble learning based on sample correlation and dynamic/semi-dynamic mechanism

import numpy as np
import re
import math
import random
import difflib
from random import sample
from keras.models import load_model


############################# Available for User Adjustment ##################################
# Load test data
# Here recommends the number of miRNAs and lncRNAs should not be too large, otherwise it will consume a lot of time.
miRNA = open('Examples\\miRNA.fasta','r').readlines() # path of miRNA file
lncRNA = open('Examples\\lncRNA.fasta','r').readlines() # path of lncRNA file
# Setting dynamic/semi-dynamic mechanism.
# Here recommends to set "dynamic" for dataset with various species samples and "semi-dynamic" for dataset with single species samples.
method = 'semi-dynamic' # can be set "dynamic" or "semi-dynamic"
############################# Available for User Adjustment ##################################

# Setting parameter
numtr = 500 # number of training representatives
# Calculate number of RNAs
nummiRNA, numlncRNA = int(len(miRNA)/2), int(len(lncRNA)/2) # number of miRNAs and lncRNAs
numts = nummiRNA * numlncRNA  # number of test samples
print('################ The number of miRNAs is ' + str(nummiRNA) + ' ################')
print('################ The number of lncRNAs is ' + str(numlncRNA) + ' ################')
print('################ The number of test samples is ' + str(numts) + ' ################')
print('################ Using the ' + method + ' method ################\n')
# Load training data
TD1MR = np.load('MR\\TD1MR.npy')
TD2MR = np.load('MR\\TD2MR.npy')
TD3MR = np.load('MR\\TD3MR.npy')
TD4MR = np.load('MR\\TD4MR.npy')
TD5MR = np.load('MR\\TD5MR.npy')
TD6MR = np.load('MR\\TD6MR.npy')
TD7MR = np.load('MR\\TD7MR.npy')
TD8MR = np.load('MR\\TD8MR.npy')
TD1PCC = np.load('PCC\\TD1PCC.npy')
TD2PCC = np.load('PCC\\TD2PCC.npy')
TD3PCC = np.load('PCC\\TD3PCC.npy')
TD4PCC = np.load('PCC\\TD4PCC.npy')
TD5PCC = np.load('PCC\\TD5PCC.npy')
TD6PCC = np.load('PCC\\TD6PCC.npy')
TD7PCC = np.load('PCC\\TD7PCC.npy')
TD8PCC = np.load('PCC\\TD8PCC.npy')
# Transfer numpy to list format
ListTD1MR = TD1MR.tolist()
ListTD2MR = TD2MR.tolist()
ListTD3MR = TD3MR.tolist()
ListTD4MR = TD4MR.tolist()
ListTD5MR = TD5MR.tolist()
ListTD6MR = TD6MR.tolist()
ListTD7MR = TD7MR.tolist()
ListTD8MR = TD8MR.tolist()
ListTD1PCC = TD1PCC.tolist()
ListTD2PCC = TD2PCC.tolist()
ListTD3PCC = TD3PCC.tolist()
ListTD4PCC = TD4PCC.tolist()
ListTD5PCC = TD5PCC.tolist()
ListTD6PCC = TD6PCC.tolist()
ListTD7PCC = TD7PCC.tolist()
ListTD8PCC = TD8PCC.tolist()
print('################ Training data loading completed ################\n')

# Pearson Correlation Coefficient Calculation
def PCCCalculateion(A, B):
    n = len(A)
    #simple sums
    sum1 = sum(float(A[i]) for i in range(n))
    sum2 = sum(float(B[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in A])
    sum2_pow = sum([pow(v, 2.0) for v in B])
    #sum up the products
    p_sum = sum([A[i]*B[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

# Matching Ratio Calculation
def MRCalculateion(A, B):
    return difflib.SequenceMatcher(None, A, B).ratio() * 1000

# kmer Extraction
def kmerExtraction(sequence, totalk):

    character = 'ATCG'
    kmer = [] # 特征

    for k in range(totalk):

        # 由于k从0开始计数，而k-mer从1开始提取，所以引入kk
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalk - kk))

        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * sequence.count(s1) / sk
                # string1 = str(f1) + ' '
                kmer.append(f1)

        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0 # 计数器
                    for lkmer2 in range(len(sequence) - kk + 1):
                        if sequence[lkmer2] == s2[0] and sequence[lkmer2 + 1] ==s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    # string2 = str(f2) + ' '
                    kmer.append(f2)

        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0 # 计数器
                        for lkmer3 in range(len(sequence) - kk + 1):
                            if sequence[lkmer3] == s3[0] and sequence[lkmer3 + 1] == s3[1] and sequence[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        # string3 = str(f3) + ' '
                        kmer.append(f3)

        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0 # 计数器
                            for lkmer4 in range(len(sequence) - kk + 1):
                                if sequence[lkmer4] == s4[0] and sequence[lkmer4 + 1] == s4[1] and sequence[lkmer4 + 2] == s4[2] and sequence[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            # string4 = str(f4) + ' '
                            kmer.append(f4)

        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0  # 计数器
                                for lkmer5 in range(len(sequence) - kk + 1):
                                    if sequence[lkmer5] == s5[0] and sequence[lkmer5 + 1] == s5[1] and sequence[lkmer5 + 2] == s5[2] and sequence[lkmer5 + 3] == s5[3] and sequence[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                # string5 = str(f5) + ' '
                                kmer.append(f5)

        # 6-mer
        if kk == 6:
            for char61 in character:
                for char62 in character:
                    for char63 in character:
                        for char64 in character:
                            for char65 in character:
                                for char66 in character:
                                    s6 = char61 + char62 + char63 + char64 + char65 + char66
                                    numkmer6 = 0  # 计数器
                                    for lkmer6 in range(len(sequence) - kk + 1):
                                        if sequence[lkmer6] == s6[0] and sequence[lkmer6 + 1] == s6[1] and sequence[lkmer6 + 2] == s6[2] and sequence[lkmer6 + 3] == s6[3] and sequence[lkmer6 + 4] == s6[4] and sequence[lkmer6 + 5] == s6[5]:
                                            numkmer6 = numkmer6 + 1
                                    f6 = wk * numkmer6 / sk
                                    # string6 = str(f6) + ' '
                                    kmer.append(f6)

    return kmer

# SSM Extraction
def SSMExtraction(sequence, totalSSM):
    character = 'ATCG'
    SSM = [] # 特征

    for k in range(totalSSM):

        # 由于k从0开始计数，而短序列从1开始提取，所以引入kk
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalSSM - kk))

        if kk == 1:
            for char11 in character:
                for char12 in character:
                    num1 = 0  # 短序列模体计数
                    for l1 in range(len(sequence) - kk - 1):
                        if sequence[l1] == char11 and sequence[l1 + kk + 1] == char12:
                            num1 = num1 + 1
                    f1 = wk * num1 / sk
                    # string1 = str(f1) + ' '
                    SSM.append(f1)

        if kk == 2:
            for char21 in character:
                for char22 in character:
                    num2 = 0  # 短序列模体计数
                    for l2 in range(len(sequence) - kk - 1):
                        if sequence[l2] == char21 and sequence[l2 + kk + 1] == char22:
                            num2 = num2 + 1
                    f2 = wk * num2 / sk
                    # string2 = str(f2) + ' '
                    SSM.append(f2)

        if kk == 3:
            for char31 in character:
                for char32 in character:
                    num3 = 0  # 短序列模体计数
                    for l3 in range(len(sequence) - kk - 1):
                        if sequence[l3] == char31 and sequence[l3 + kk + 1] == char32:
                            num3 = num3 + 1
                    f3 = wk * num3 / sk
                    # string3 = str(f3) + ' '
                    SSM.append(f3)

        if kk == 4:
            for char41 in character:
                for char42 in character:
                    num4 = 0  # 短序列模体计数
                    for l4 in range(len(sequence) - kk - 1):
                        if sequence[l4] == char41 and sequence[l4 + kk + 1] == char42:
                            num4 = num4 + 1
                    f4 = wk * num4 / sk
                    # string4 = str(f4) + ' '
                    SSM.append(f4)

        if kk == 5:
            for char51 in character:
                for char52 in character:
                    num5 = 0  # 短序列模体计数
                    for l5 in range(len(sequence) - kk - 1):
                        if sequence[l5] == char51 and sequence[l5 + kk + 1] == char52:
                            num5 = num5 + 1
                    f5 = wk * num5 / sk
                    # string5 = str(f5) + ' '
                    SSM.append(f5)

    return SSM

# Complex feature construction
def CFConstruction(feature1, feature2):
    CF = []
    for i in range(len(feature1)):
        a = feature1[i]
        for j in range(len(feature2)):
            b = feature2[j]
            c = (a + b) / 2    # AM算术平均数复杂特征
            CF.append(c)
    return CF

# Data Preprocessing
def DataPreprocessing(ListmiRNA, ListlncRNA):
    Data, Sequence, Feature = [], [], []
    for indmiRNA in range(len(ListmiRNA)):
        LinemiRNA = ListmiRNA[indmiRNA]
        if '>' in LinemiRNA:
            miRNAname = LinemiRNA.strip()
            miRNAsequence = ListmiRNA[indmiRNA + 1].strip()
            miRNAsequence = miRNAsequence.replace('U', 'T')

            for indlncRNA in range(len(ListlncRNA)):
                LinelncRNA = ListlncRNA[indlncRNA]
                if '>' in LinelncRNA:
                    lncRNAname = LinelncRNA.strip()
                    lncRNAname = lncRNAname[1:]
                    lncRNAsequence = ListlncRNA[indlncRNA + 1].strip()

                    miRNAkmer = kmerExtraction(miRNAsequence, 3)
                    lncRNAkmer = kmerExtraction(lncRNAsequence, 3)
                    miRNASSM = SSMExtraction(miRNAsequence, 3)
                    lncRNASSM = SSMExtraction(lncRNAsequence, 3)

                    PairData = miRNAname + ',' + lncRNAname + ',' + miRNAsequence + ',' + lncRNAsequence
                    PairSequence = miRNAsequence + lncRNAsequence
                    PairFeature = miRNAkmer + lncRNAkmer + miRNASSM + lncRNASSM

                    Data.append(PairData)
                    Sequence.append(PairSequence)
                    Feature.append(PairFeature)

    return Data, Sequence, Feature

# Data Conversion
def DataConversion(TestComFea, Frow, Fcolumn):
    X = []
    NumberColumn = len(TestComFea[0])
    for Sample in TestComFea:
        Feature = Sample[0 : NumberColumn]
        FeatureForm = np.array(Feature).astype('float32').reshape(-1, Frow) # 更改特征格式
        X.append(FeatureForm)
    X = np.array(X).reshape(-1, Fcolumn, Frow, 1)
    return X

# Indicator Calculation
def IndicatorCalculation(AS, AF, numtr, TD1MRSample, TD1PCCSample, TD2MRSample, TD2PCCSample, TD3MRSample, TD3PCCSample, TD4MRSample, TD4PCCSample,
                         TD5MRSample, TD5PCCSample, TD6MRSample, TD6PCCSample, TD7MRSample, TD7PCCSample, TD8MRSample, TD8PCCSample):
    mrim = np.zeros((numtr, 8))  # MR indicator matrix
    pccim = np.zeros((numtr, 8))  # PCC indicator matrix
    for indtrain in range(numtr):
        BS1 = TD1MRSample[indtrain]
        BF1 = TD1PCCSample[indtrain]
        mr1 = MRCalculateion(AS, BS1)
        pcc1 = PCCCalculateion(AF, BF1)
        mrim[indtrain][0] = mr1
        pccim[indtrain][0] = pcc1
        BS2 = TD2MRSample[indtrain]
        BF2 = TD2PCCSample[indtrain]
        mr2 = MRCalculateion(AS, BS2)
        pcc2 = PCCCalculateion(AF, BF2)
        mrim[indtrain][1] = mr2
        pccim[indtrain][1] = pcc2
        BS3 = TD3MRSample[indtrain]
        BF3 = TD3PCCSample[indtrain]
        mr3 = MRCalculateion(AS, BS3)
        pcc3 = PCCCalculateion(AF, BF3)
        mrim[indtrain][2] = mr3
        pccim[indtrain][2] = pcc3
        BS4 = TD4MRSample[indtrain]
        BF4 = TD4PCCSample[indtrain]
        mr4 = MRCalculateion(AS, BS4)
        pcc4 = PCCCalculateion(AF, BF4)
        mrim[indtrain][3] = mr4
        pccim[indtrain][3] = pcc4
        BS5 = TD5MRSample[indtrain]
        BF5 = TD5PCCSample[indtrain]
        mr5 = MRCalculateion(AS, BS5)
        pcc5 = PCCCalculateion(AF, BF5)
        mrim[indtrain][4] = mr5
        pccim[indtrain][4] = pcc5
        BS6 = TD6MRSample[indtrain]
        BF6 = TD6PCCSample[indtrain]
        mr6 = MRCalculateion(AS, BS6)
        pcc6 = PCCCalculateion(AF, BF6)
        mrim[indtrain][5] = mr6
        pccim[indtrain][5] = pcc6
        BS7 = TD7MRSample[indtrain]
        BF7 = TD7PCCSample[indtrain]
        mr7 = MRCalculateion(AS, BS7)
        pcc7 = PCCCalculateion(AF, BF7)
        mrim[indtrain][6] = mr7
        pccim[indtrain][6] = pcc7
        BS8 = TD8MRSample[indtrain]
        BF8 = TD8PCCSample[indtrain]
        mr8 = MRCalculateion(AS, BS8)
        pcc8 = PCCCalculateion(AF, BF8)
        mrim[indtrain][7] = mr8
        pccim[indtrain][7] = pcc8
    return mrim, pccim

############################### Main Process #####################################
# Preprocess test data
Data, Sequence, Feature = DataPreprocessing(miRNA, lncRNA)
print('################ Data preprocessing completed ################\n')

# Construct and fuse complex feature
TestComFea = []
for indcf in range(numts):
    TestSample = Data[indcf]
    miRNAname, lncRNAname, miRNAsequence, lncRNAsequence = TestSample.split(',')
    miRNAkmer = kmerExtraction(miRNAsequence, 3) # miRNA kmer
    miRNASSM = SSMExtraction(miRNAsequence, 3) # miRNA SSM
    lncRNAkmer = kmerExtraction(lncRNAsequence, 3) # lncRNA kmer
    lncRNASSM = SSMExtraction(lncRNAsequence, 3) # lncRNA SSM
    cfkmer = CFConstruction(miRNAkmer, lncRNAkmer) # kmer complex feature construction
    cfSSM = CFConstruction(miRNASSM, lncRNASSM) # SSM complex feature construction
    cffusion = cfkmer + cfSSM # complex feature fusion
    TestComFea.append(cffusion)
print('################ Complex feature construction and fusion completed ################\n')

# Load models
Model1 = load_model('Base models\\Model1.h5')
Model2 = load_model('Base models\\Model2.h5')
Model3 = load_model('Base models\\Model3.h5')
Model4 = load_model('Base models\\Model4.h5')
Model5 = load_model('Base models\\Model5.h5')
Model6 = load_model('Base models\\Model6.h5')
Model7 = load_model('Base models\\Model7.h5')
Model8 = load_model('Base models\\Model8.h5')
print('################ Model loading completed ################\n')

# Evaluate test data parallelly
X_TestComFea = DataConversion(TestComFea, 3, 3120) # Conversion of complex feature vector to matrix
EvaluationMatrix = np.zeros((numts, 8)) # Evaluation matrix
EvaluationMatrix[:, 0] = Model1.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 1] = Model2.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 2] = Model3.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 3] = Model4.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 4] = Model5.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 5] = Model6.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 6] = Model7.predict(X_TestComFea)[:, 1]
EvaluationMatrix[:, 7] = Model8.predict(X_TestComFea)[:, 1]
print('################ Model evaluation completed ################\n')

# Choose 500 training representatives randomly
TD1MRSample = sample(ListTD1MR, numtr)
TD2MRSample = sample(ListTD2MR, numtr)
TD3MRSample = sample(ListTD3MR, numtr)
TD4MRSample = sample(ListTD4MR, numtr)
TD5MRSample = sample(ListTD5MR, numtr)
TD6MRSample = sample(ListTD6MR, numtr)
TD7MRSample = sample(ListTD7MR, numtr)
TD8MRSample = sample(ListTD8MR, numtr)
TD1PCCSample = sample(ListTD1PCC, numtr)
TD2PCCSample = sample(ListTD2PCC, numtr)
TD3PCCSample = sample(ListTD3PCC, numtr)
TD4PCCSample = sample(ListTD4PCC, numtr)
TD5PCCSample = sample(ListTD5PCC, numtr)
TD6PCCSample = sample(ListTD6PCC, numtr)
TD7PCCSample = sample(ListTD7PCC, numtr)
TD8PCCSample = sample(ListTD8PCC, numtr)
print('################ Training representatives selection completed ################\n')

# Calculate indicators between training representatives and test samples
MRIM = np.zeros((numts + 1, 8)) # MR indicator matrix
PCCIM = np.zeros((numts + 1, 8)) # PCC indicator matrix
for indtest in range(numts):
    AS = Sequence[indtest]
    AF = Feature[indtest]
    mrim, pccim = IndicatorCalculation(AS, AF, numtr, TD1MRSample, TD1PCCSample, TD2MRSample, TD2PCCSample, TD3MRSample, TD3PCCSample, TD4MRSample, TD4PCCSample,
                                       TD5MRSample, TD5PCCSample, TD6MRSample, TD6PCCSample, TD7MRSample, TD7PCCSample, TD8MRSample, TD8PCCSample)

    MRIM[indtest][:] = np.mean(mrim, axis=0)
    PCCIM[indtest][:] = np.mean(pccim, axis=0)
    print('################ Indicators calculation ' + str(indtest + 1) + '/' + str(numts) + ' ################')
MRIM[numts][:] = np.mean(MRIM, axis=0)
PCCIM[numts][:] = np.mean(PCCIM, axis=0)
print('################ Indicator calculation completed ################\n')

# Obtain selection matrix
SelectionMatrix = np.zeros((numts, 8)) # Selection matrix
avgmr = np.mean(MRIM, axis=1) # average rule for MR
avgpcc = np.mean(PCCIM, axis=1) # average rule for PCC
if method == 'dynamic':
    for row in range(numts):
        for column in range(8):
            if column == 2: # Model3 is selected using MR indicator
                if MRIM[row][column] >= avgmr[row]:
                    SelectionMatrix[row][column] = 1
            else: # other base models are selected using PCC indicator
                if PCCIM[row][column] >= avgpcc[row]:
                    SelectionMatrix[row][column] = 1
if method == 'semi-dynamic':
    for column in range(8):
        if column == 2: # Model3 is selected using MR indicator
            if MRIM[numts][column] >= avgmr[numts]:
                SelectionMatrix[:, column] = 1
        else: # other base models are selected using PCC indicator
            if PCCIM[numts][column] >= avgpcc[numts]:
                SelectionMatrix[:, column] = 1
print('################ Model selection completed ################\n')

# Output result
ResultMatrix = EvaluationMatrix * SelectionMatrix
Result = np.sum(ResultMatrix, axis=1) / np.sum(SelectionMatrix, axis=1)
Output = []
for indresult in range(numts):
    TestSample2 = Data[indresult]
    miRNAname2, lncRNAname2, miRNAsequence2, lncRNAsequence2 = TestSample2.split(',')
    if Result[indresult] >= 0.5:
        StrResult = miRNAname2 + ',' + lncRNAname2 + ',' + 'Interaction' + '\n'
    else:
        StrResult = miRNAname2 + ',' + lncRNAname2 + ',' + 'Non-Interaction' + '\n'
    Output.append(StrResult)
print('The predicted results have been output!')
w = open('ASELSD_Output_' + method + '.fasta', 'w')
w.writelines(Output)
w.close()
