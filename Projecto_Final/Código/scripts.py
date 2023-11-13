import pickle
import re

def openFile(filename):
    return pickle.load(open(filename, 'rb'))

def clearText(documentos, binary=False):

    if binary:
        documentos = [doc.decode('UTF-8') for doc in documentos]

    documentos = [doc.replace('<br />', ' ') for doc in documentos]
    documentos = [re.sub(r'[^a-zA-Z\u00C0-\u00FF]+' , ' ', doc) for doc in documentos]
    
    return documentos


def text2vector(Docs, tipo_classificacao, dir, binary=False):
    Docs  = clearText(Docs, binary)
    tfidf = None

    #se binaria    
    if tipo_classificacao   == 'b':
        tfidf = openFile(dir)['tfidf']
    #se multiclasse
    elif tipo_classificacao == 'm':
        tfidf = openFile(dir)['tfidf']
    
    #utilizar modelo armazenado no pickle para transformar X
    if (tfidf):
        X = tfidf.transform(Docs)
        return X

def binClassify(X, clas, svd, dir):
    #se regressao logistica
    if clas == 'logReg':
        #se houver truncated
        if svd == True:
            classificador = pickle.load(open(dir, 'rb'))['logReg']
        #sem truncated
        elif svd == False:
            classificador = pickle.load(open(dir, 'rb'))['logReg']
     
    #se linearSVC
    elif clas == 'SVC':
        classificador = pickle.load(open(dir, 'rb'))['lSVC']

    #se k-neigh
    elif clas == 'knn':
        classificador = pickle.load(open(dir, 'rb'))['knn']

    #utilizar modelo armazenado no pickle para avaliar dados
    return classificador.predict(X)

def multiClassify(X, clas, svd, dir):
    #se regressao logistica
    if clas == 'logReg':
        #com truncated
        if svd == True:
            classificador = pickle.load(open(dir, 'rb'))['logReg']
        #sem truncated
        elif svd == False:
            classificador = pickle.load(open(dir, 'rb'))['logReg']

    #se linearSVC
    elif clas == 'SVC':
        classificador = pickle.load(open(dir, 'rb'))['lSVC']

    #se k-neigh
    elif clas == 'knn':
        classificador = pickle.load(open(dir, 'rb'))['knn']

    #utilizar modelo armazenado no pickle para avaliar dados
    return classificador.predict(X)

