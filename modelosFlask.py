import numpy as np
import pickle
from flask import Flask,request,json

scalerD = pickle.load(open( 'scalerD.p', 'rb' ))
scalerDp = pickle.load(open( 'scalerDp.p', 'rb' ))
scalerND = pickle.load(open( 'scalerND.p', 'rb' ))
scalerNDp = pickle.load(open( 'scalerNDp.p', 'rb' ))
pca = pickle.load(open( 'pca.p', 'rb' ))

MmodeloKNN = pickle.load(open('knn.p','rb'))
MmodeloKNNp = pickle.load(open('knnp.p','rb'))
MmodeloNN = pickle.load(open('model.p','rb'))
MmodeloNNp = pickle.load(open('modelp.p','rb'))
MmodeloSVMr = pickle.load(open('modeloSVMr.p','rb'))
MmodeloSVMrp = pickle.load(open('modeloSVMrp.p','rb'))
MmodeloSVMl = pickle.load(open('modeloSVMl.p','rb'))
MmodeloSVMlp = pickle.load(open('modeloSVMlp.p','rb'))
MmodeloSVMp = pickle.load(open('modeloSVMp.p','rb'))
MmodeloSVMpp = pickle.load(open('modeloSVMpp.p','rb'))
MmodeloSVMs = pickle.load(open('modeloSVMs.p','rb'))
MmodeloSVMsp = pickle.load(open('modeloSVMsp.p','rb'))

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def holaMundo():
    return 'Aqui'

@app.route('/modeloKNNsinPCA',methods=['POST'])
def modeloKNN():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=MmodeloKNN.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloKNNconPCA',methods=['POST'])
def modeloKNNp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=MmodeloKNNp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloNNsinPCA',methods=['POST'])
def modeloNN():
    content = request.get_json()
    X = getX(content,10)
    X = scalerD.transform(X)
    predict=MmodeloNN.predict_classes(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloNNconPCA',methods=['POST'])
def modeloNNp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerDp.transform(X)
    predict=MmodeloNNp.predict_classes(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMrsinPCA',methods=['POST'])
def modeloSVMr():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=MmodeloSVMr.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMrconPCA',methods=['POST'])
def modeloSVMrp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=MmodeloSVMrp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMlsinPCA',methods=['POST'])
def modeloSVMl():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=MmodeloSVMl.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMlconPCA',methods=['POST'])
def modeloSVMlp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=MmodeloSVMlp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMpsinPCA',methods=['POST'])
def modeloSVMp():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=MmodeloSVMp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMpconPCA',methods=['POST'])
def modeloSVMpp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=MmodeloSVMpp.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMssinPCA',methods=['POST'])
def modeloSVMs():
    content = request.get_json()
    X = getX(content,10)
    X = scalerND.transform(X)
    predict=MmodeloSVMs.predict(X)
    response = jsonGen(predict[0])
    return response

@app.route('/modeloSVMsconPCA',methods=['POST'])
def modeloSVMsp():
    content = request.get_json()
    X = getX(content,10)
    X = pca.transform(X)
    X = scalerNDp.transform(X)
    predict=MmodeloSVMsp.predict(X)
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
    app.run()