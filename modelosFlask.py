import numpy as np
import pickle
from flask import Flask,request,json

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def holaMundo():
    return 'Aqui'

@app.route('/modeloKNNsinPCA',methods=['POST'])
def modeloKNN():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=modeloKNN.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloKNNconPCA',methods=['POST'])
def modeloKNNp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=modeloKNNp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloNNsinPCA',methods=['POST'])
def modeloNN():
    content = request.get_json()
    X = getX(content,10)
    X = scalerD.transform(X)
    predict=modeloNN.predict_classes(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloNNconPCA',methods=['POST'])
def modeloNNp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerDp.transform(X)
    predict=modeloNNp.predict_classes(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMrsinPCA',methods=['POST'])
def modeloSVMr():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=modeloSVMr.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMrconPCA',methods=['POST'])
def modeloSVMrp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=modeloSVMrp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMlsinPCA',methods=['POST'])
def modeloSVMl():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=modeloSVMl.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMlconPCA',methods=['POST'])
def modeloSVMlp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=modeloSVMlp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMpsinPCA',methods=['POST'])
def modeloSVMp():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=modeloSVMp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMpconPCA',methods=['POST'])
def modeloSVMpp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=modeloSVMpp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMssinPCA',methods=['POST'])
def modeloSVMs():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=modeloSVMs.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMsconPCA',methods=['POST'])
def modeloSVMsp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=modeloSVMsp.predict(X)
    response = jsonGen(predict[0])
    return response

def getX(content,n):
    X=[]
    for i in range(0,n):
        X=np.append(X,content['var'+str(i+1)])
    X = np.atleast_2d(X)
    return X

def jsonGen(predict):
    d={'prediccion':str(predict)}
    response = app.response_class(
        response = json.dumps(d),
        status = 200,
        mimetype = 'application/json'
    )
    return response

if __name__ == '__main__':
    scalerD = pickle.load(open( 'scalerD.p', 'rb' ))
    scalerDp = pickle.load(open( 'scalerDp.p', 'rb' ))
    scalerND = pickle.load(open( 'scalerND.p', 'rb' ))
    scalerNDp = pickle.load(open( 'scalerNDp.p', 'rb' ))
    pca = pickle.load(open( 'pca.p', 'rb' ))
    modeloKNN = pickle.load(open('knn.p','rb'))
    modeloKNNp = pickle.load(open('knnp.p','rb'))
    modeloNN = pickle.load(open('model.p','rb'))
    modeloNNp = pickle.load(open('modelp.p','rb'))
    modeloSVMr = pickle.load(open('modeloSVMr.p','rb'))
    modeloSVMrp = pickle.load(open('modeloSVMrp.p','rb'))
    modeloSVMl = pickle.load(open('modeloSVMl.p','rb'))
    modeloSVMlp = pickle.load(open('modeloSVMlp.p','rb'))
    modeloSVMp = pickle.load(open('modeloSVMp.p','rb'))
    modeloSVMpp = pickle.load(open('modeloSVMpp.p','rb'))
    modeloSVMs = pickle.load(open('modeloSVMs.p','rb'))
    modeloSVMsp = pickle.load(open('modeloSVMsp.p','rb'))
    app.run()