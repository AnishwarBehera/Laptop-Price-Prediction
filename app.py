from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
app.static_folder = 'static'
app.static_url_path = '/static'

def ppi(x,screenSize):
    resolution={
        'HD':[1280,720],
        'FullHD':[1920,1080],
        'QuadHD':[2560,1440],
        '4K':[3840,2160]
    }
    lst=resolution[x]
    return ((lst[0]**2+lst[1]**2)**0.5)/screenSize


model = pickle.load(open('pipe.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate',methods=['POST'])
def predict_diabetes():
    manufacturer = request.form.get('manufacturer')
    Category = request.form.get('Category')
    screenSize = float(request.form.get('screenSize'))
    processor = request.form.get('processor')
    memory = int(request.form.get('memory'))
    graphicsCard = request.form.get('graphicsCard')
    operatingSystem = request.form.get('operatingSystem')
    weight = float(request.form.get('weight'))
    touchscreen = int(request.form.get('touchscreen'))
    ipsScreen = int(request.form.get('ipsScreen'))
    screenResolution = request.form.get('screenResolution')
    ssd = int(request.form.get('ssd'))
    hdd = int(request.form.get('hdd'))
    flashStorage = int(request.form.get('flashStorage'))


    PPI=round(ppi(screenResolution,screenSize),2)


    dd = pd.DataFrame([[manufacturer,Category,screenSize,processor,memory,graphicsCard,operatingSystem,weight,touchscreen,ipsScreen,PPI,ssd,hdd,flashStorage]],
                      columns=['Manufacturer', 'Category','Screen Size', 'CPU', 'RAM', 'GPU','Operating System','Weight', 'Touchscreen','Ips', 'PPI','SSD','HDD','Flash_Storage'])
    
    result = model.predict(dd)
    result=round(int(np.exp(result)[0])/178)
    result=(f'The estimated price would be between {result-2500} and {result+2500}')
    
    return render_template('result.html',result=result)

if __name__=='__main__':
    # app.run(host='0.0.0.0')
    app.run(host='0.0.0.0',port=8080)


