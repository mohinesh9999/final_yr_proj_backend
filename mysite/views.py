from copy import copy
from pymongo import MongoClient  #for mongodb
import jwt    #for authentication
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import math,random
from django.core.mail import send_mail
import datetime;
#connecting python with mongodb
client=MongoClient("mongodb+srv://test:test@cluster0-nc9ml.mongodb.net/sih?retryWrites=true&w=majority")
db=client.get_database('sih')
record=db.sih
from rest_framework.decorators import api_view
import hashlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#buffer conversion
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse,JsonResponse
import base64
import PIL, PIL.Image
from io import StringIO
import io

d=os.path.dirname(os.getcwd())
# d=os.path.join(d,"mysite")
d=os.path.join(d,"app")
d=os.path.join(d,"sih")
xn=d
d=os.path.join(d,"States")



def test(request):
    return JsonResponse({'test':'pass'},status=200)
def generateOTP(): 
    digits = "0123456789"
    OTP = "" 
    for i in range(4) : 
        OTP += digits[math.floor(random.random() * 10)] 
    return OTP
def sendMail(to,otp):
    fromaddr = "sihkkr2020@gmail.com"
    toaddr = to
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "SUBJECT OF THE MAIL"

    body = otp
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "demon_killers")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
@api_view(['POST'])
def signup(request):
    # print((request))
    if request.method == "POST":
        try:
            # print((request),request.data)
            x=record.insert_one(dict(request.data))
            return JsonResponse({"status": dict(request.data)},status=200)
        except Exception as e:
            return JsonResponse({"status": "email already exist"},status=500)
    else:
        return JsonResponse({"status": "Only post method allowed"},status=500)

@api_view(['POST'])
def sendEmail(request):
    q=request.data
    x=record.find_one({"_id":q['email']})
    if(x==None):
        try:
            otp=generateOTP()
            print(request.data['email'],otp)
            # sendEmail(request.data['email'],otp)
            send_mail('Verificaton for signup', otp, 'sihkkr2020@gmail.com', [request.data['email']])
            return JsonResponse({"status":hashlib.md5(otp.encode()).hexdigest()},status=200)
        except Exception as e:
            return JsonResponse({"status": "an error occured :(","e":e},status=500)
    else:
        return JsonResponse({"status":'already registered'},status=200)


@api_view(['POST'])
def sendEmailFP(request):
    q=request.data
    x=record.find_one({"_id":q['email']})
    if(x!=None):
        try:
            otp=generateOTP()
            print(request.data['email'],otp)
            # sendEmail(request.data['email'],otp)
            send_mail('Verificaton for signup', otp, 'sihkkr2020@gmail.com', [request.data['email']])
            return JsonResponse({"status":hashlib.md5(otp.encode()).hexdigest()},status=200)
        except Exception as e:
            return JsonResponse({"status": "an error occured :(","e":e},status=500)
    else:
        return JsonResponse({"status":'not registerd'},status=200)


@api_view(['POST'])
def FP(request):
    q=request.data
    record.update_many( {"_id":q['email']}, { "$set":{  "password":q['password']} } ) 
    return JsonResponse({"status":'done'},status=200)



@api_view(['POST'])
def Query(request):
    q=request.data
    y=request.data['token']
    y=jwt.decode(y, 'mks', algorithms=['HS256'])
    y1=record.find_one({"_id":y['email']})
    z=y1['query']
    z.append([q['msg'],q['name'],q['email'],datetime.datetime.now().isoformat()])
    record.update_many( {"_id":y['email']}, { "$set":{  "query":z} } ) 
    return JsonResponse({"status":'done'},status=200)





@api_view(['POST'])
def login(request):
    try:
        q=request.data
        print(q)
        y=jwt.encode({"email":q['email']},"mks")
        x=record.find_one({"_id":q['email'],"password":q['password']})
        #print(y.decode('UTF-8'),x,q,jwt.decode(y, 'mks', algorithms=['HS256']))
        if(x!=None):
            return JsonResponse({"status": "True","token":y},status=200)
        else:
            return JsonResponse({"status": "False"},status=200)
    except Exception as e:
        return JsonResponse({"status": "an error occured :(","e":e},status=500)











@api_view(['POST'])
def getUserDetails(request):
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks', algorithms=['HS256'])
        y=record.find_one({"_id":y['email']})
        return JsonResponse({"status": "True","details":y},status=200)
    except Exception as e:
        return JsonResponse({"status": "an error occured :(","e":e},status=500)








@api_view(['POST'])
def mlModel(request):
    global d,xn
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks', algorithms=['HS256'])
        y=record.find_one({"_id":y['email']})
        z=y['recent']
        z.append([request.data['state'],request.data['city'],request.data['month'],datetime.datetime.now().isoformat()])
        print(y,z)
        record.update_many( {"_id":y['_id']}, { "$set":{  "recent":z} } ) 
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,request.data['state'])
        q=os.path.join(e,request.data['city'].capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[request.data['month']]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[request.data['month']]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        plt.xticks(xi, x)
        plt.legend
        plt.show()
        l=[]
        for i in y_pred:
            l.append(i)
        return JsonResponse({"buffer":l,"flag":'True'})
    except Exception as e:
        return JsonResponse({"status": False,"flag":'False'},status=400)
@api_view(['POST'])
def mlModel1(request):
    global d,xn
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks', algorithms=['HS256'])
        
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,request.data['state'])
        q=os.path.join(e,request.data['city'].capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[request.data['month']]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[request.data['month']]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        plt.xticks(xi, x)
        plt.legend
        plt.show()
        l=[]
        for i in y_pred:
            l.append(i)
        w=[]
        msp=5550
        for i in y_pred:
            if(msp>=i):
                w.append(msp)
            else:
                w.append(msp+0.4*(i-msp))
        return JsonResponse({"buffer":l,'w':w,'flag':'True'})
    except Exception as e:
        print(str(e))
        return JsonResponse({"status": False,"flag":'False'},status=400)
@api_view(['POST'])
def mlModel2(request):
    global d,xn
    try:
        e=copy(d)
        y=request.data['token']
        y=jwt.decode(y, 'mks', algorithms=['HS256'])
        import matplotlib
        matplotlib.use('Agg')
        e=os.path.join(d,request.data['state'])
        e=os.path.join(e,request.data['city'].capitalize())
        os.chdir(e)
        dataset = pd.read_csv('real.csv')
        # os.chdir(e)
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        indexedDataset = dataset.set_index(['Date'])
        indexedDataset = indexedDataset.fillna(method='ffill')


        from datetime import datetime
        #indexedDataset.tail(12)



        rolmean = indexedDataset.rolling(window=12).mean()

        rolstd = indexedDataset.rolling(window=12).std()
        #print(rolmean,rolstd)



        from statsmodels.tsa.stattools import adfuller

        #print('Results of DFT: ')
        dftest = adfuller(indexedDataset['Prices'],autolag='AIC')

        dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-val','lag used','Number of obser'])
        

        indexedDataset_logScale=np.log(indexedDataset)

        movingAverage=indexedDataset_logScale.rolling(window=12).mean()
        movingstd=indexedDataset_logScale.rolling(window=12).std()


        datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
        #datasetLogScaleMinusMovingAverage.head(12)

        datasetLogScaleMinusMovingAverage.dropna(inplace=True)
        #datasetLogScaleMinusMovingAverage.head(12)

        from statsmodels.tsa.stattools import adfuller
        def test_stationarity(timeseries):
            
            movingAverage=timeseries.rolling(window=12).mean()
            movingSTD=timeseries.rolling(window=12).std()
            
            dftest=adfuller(timeseries['Prices'],autolag='AIC')
            dfoutput=pd.Series(dftest[0:4],index=['Test stats','pval','lag','No of obser'])
        

        test_stationarity(datasetLogScaleMinusMovingAverage)

        exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()


        datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
        test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

        datasetLogDiffShifting=indexedDataset_logScale - indexedDataset_logScale.shift()


        datasetLogDiffShifting.dropna(inplace=True)
        test_stationarity(datasetLogDiffShifting)





        import statsmodels.api as sm

        model= sm.tsa.arima.ARIMA(indexedDataset_logScale,order=(1,1,1))
        results_AR=model.fit()


        predictions_ARIMA_diff=pd.Series(results_AR.fittedvalues,copy=True)
        #print(predictions_ARIMA_diff.head())

        predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
        #print(predictions_ARIMA_diff_cumsum.head())

        predictions_ARIMA_log=pd.Series(indexedDataset_logScale['Prices'].iloc[0],index=indexedDataset_logScale.index)
        predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        #predictions_ARIMA_log.head()

        predictions_ARIMA=np.exp(predictions_ARIMA_log)

        #indexedDataset_logScale
        #predictions_ARIMA

        modell= sm.tsa.arima.ARIMA(predictions_ARIMA,order=(1,1,1))
        results_ARM=modell.fit()

        # results_ARM.plot_predict(1,60)
        x=results_ARM.forecast(steps=12)



        toplot=x[0][0:12]
        print(toplot,x[2])
        l=[]
        for i in toplot:
            l.append(i)
        w=[]
        for i in x[2]:
            i=list(i)
            w.append([i[0],i[1]])
        print(w)
        return JsonResponse({"buffer":l,'flag':'True',"x":w})
    except Exception as e:
        print(str(e))
        return JsonResponse({"status": False,"flag":'False'},status=400)

import asyncio
l1=[]
def regression(city,state,month):
    try:
        global l1
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,state)
        q=os.path.join(e,city.capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[month]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[month]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        # plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        # plt.xticks(xi, x)
        # plt.legend
        # plt.show()
        # l1.append(city,state,sum(list(y_pred))/(len(y_pred)))
        # import time
        # print(y_pred)
        # time.sleep(1)
        k=0
        for i in range(len(list(y_pred))):
            k+=float(y_pred[i])
        # print(city,state)
        # l1.append([city,state,(k/len(list(y_pred)))])
        # print('eeee')
        return [city,state,(k/len(list(y_pred)))]
    except Exception as e:
        print(e)
        pass
async def first():
    await asyncio.sleep(1)
    return "1"

async def second():
    await asyncio.sleep(1)
    return "2"
async def main(a):
    l=[]
    async def one_iteration(i,j,k):
        l.append(await regression(i,j,k))
    d={
                "andhrapradesh":["Kurnool"],
                "gujarat":["Ahmedabad","Amreli","Bhavnagar","Gandhinagar","Jamnagar","Junagarh","Kheda","Kutch","Rajkot"],
                "haryana":["Hisar","Jind","Sirsa"],
                "karnataka":["Bijapur","Davanagere","Dharwad","Haveri","Raichur"],
                "madhyapradesh":["Alrajpur","Badwani","Chindwara","Dhar","Khargone"],
                "maharashta":["Amravati","Aurangabad","Buldana","Nagpur","Nanded"],
                "punjab":["Barnala","Bathinda","Mansa"],
                "rajasthan":["Bhilwara","Hanumangarh","Sri Ganganagar"],
                "tamilnadu":["Tuticorin"],
                "telangana":["Khammam","Warangal"],
                "up":["Hathras" ]
            }
    coros=[]
    for i in d:
        for j in d[i]:
            coros.append(one_iteration(j,i,'january'))
    # coros = [one_iteration() for _ in range(12)]
    await asyncio.gather(*coros)
    print(coros)
    return JsonResponse({'flag':'True'})
@api_view(['POST'])
def allprice(request):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(request))
@api_view(['POST'])
def allprice1(request):
    global l1
    l=[]
    d={
                "andhrapradesh":["Kurnool"],
                "gujarat":["Ahmedabad","Amreli","Bhavnagar","Gandhinagar","Jamnagar","Junagarh","Kheda","Kutch","Rajkot"],
                "haryana":["Hisar","Jind","Sirsa"],
                "karnataka":["Bijapur","Davanagere","Dharwad","Haveri","Raichur"],
                "madhyapradesh":["Alrajpur","Badwani","Chindwara","Dhar","Khargone"],
                "maharashta":["Amravati","Aurangabad","Buldana","Nagpur","Nanded"],
                "punjab":["Barnala","Bathinda","Mansa"],
                "rajasthan":["Bhilwara","Hanumangarh","Sri Ganganagar"],
                "tamilnadu":["Tuticorin"],
                "telangana":["Khammam","Warangal"],
                "up":["Hathras" ]
            }
    x=[]
    # x=[regression(j , i,'january') for j in d[i] for i in d]
    from opencage.geocoder import OpenCageGeocode
    from geopy.distance import geodesic
    def find_distance(A,B):
        key = 'c1fedb389d6f4101aa6f45e2c8e518d2'  # get api key from:  https://opencagedata.com
        geocoder = OpenCageGeocode(key)
        
        result_A = geocoder.geocode(A)
        lat_A = result_A[0]['geometry']['lat']
        lng_A = result_A[0]['geometry']['lng']
        
        result_B = geocoder.geocode(B)
        lat_B = result_B[0]['geometry']['lat']
        lng_B = result_B[0]['geometry']['lng']  
        
        return (geodesic((lat_A,lng_A), (lat_B,lng_B)).kilometers)
    for i in d:
        for j in d[i]:
            m=regression(j , i,request.data['month'])
            # w=find_distance('gwalior',j)
            av_price=3
            if(m):
                try:
                    m.append(av_price*find_distance(request.data['city'],j))
                    m.append(m[-1]+m[-2])
                    x.append(m)
                except:
                    pass
                # print()
            # print(i,j,m)av_price
    # print(l1)

     
    # def main():
    #     find_distance("kurukshetra","gwalior")
    # main()
    from copy import copy
    x1=copy(x)
    x.sort(key=lambda x:x[2])
    x1.sort(key=lambda x:x[4])
    return JsonResponse({"buffer":x,"buffer1":x1,'flag':'True'})
@api_view(['GET'])
def allprice2(request):
    try:
        from opencage.geocoder import OpenCageGeocode
        from geopy.distance import geodesic
        def find_distance(A,B):
            key = 'c1fedb389d6f4101aa6f45e2c8e518d2'  # get api key from:  https://opencagedata.com
            geocoder = OpenCageGeocode(key)
            
            result_A = geocoder.geocode(A)
            lat_A = result_A[0]['geometry']['lat']
            lng_A = result_A[0]['geometry']['lng']
            
            result_B = geocoder.geocode(B)
            lat_B = result_B[0]['geometry']['lat']
            lng_B = result_B[0]['geometry']['lng']  
        
            return (geodesic((lat_A,lng_A), (lat_B,lng_B)).kilometers)
        w=regression(request.data['city'],request.data['state'],request.data['month'])
        w.append(3*find_distance(request.data['city'],request.data['tcity']))
        return JsonResponse({"buffer":w,'w':'yo'})
    except Exception as e:
        return JsonResponse({"buffer":[],'e':e})
def mlmodel3():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # %matplotlib inline
    from sklearn.model_selection import train_test_split
    import warnings
    from fbprophet import Prophet
    from fbprophet.plot import plot_plotly
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns
    warnings.filterwarnings("ignore")

    df = pd.read_csv('/content/sample_data/Data_modified.csv')
    df.head(),df.shape

    df.info()

    print(min(df['Price']))
    print(max(df['Price']))

    df['Price'].describe()

    df = df[df['Price'] > 0]
    df = df[df['Price'] < 13000]

    print(min(df['Price']))
    print(max(df['Price']))

    df['Day']=df['Day'].apply(lambda x: '{0:0>2}'.format(x))
    df['Month']=df['Month'].apply(lambda x: '{0:0>2}'.format(x))
    df['Year'] = df['Year'].apply(str)
    df['Day']=df['Day'].apply(str)
    df['Month']=df['Month'].apply(str)
    df['date'] = df['Year'].str.cat(df['Month'], sep ="-")
    df['date'] = df['date'].str.cat(df['Day'], sep ="-")
    df = df.drop(['Day', 'Month', "Year"], axis=1)
    df.head()

    df['State'].unique()

    df_prop= pd.DataFrame()
    df_prop['ds'] = pd.to_datetime(df["date"])
    df_prop['y'] = df["Price"]
    df_prop['State'] = df["State"]
    df_prop['District'] = df["District"]
    df_prop['Market'] = df["Market"]
    df_prop.head()

    df_prop['State'] = df_prop['State'].astype('category')
    df_prop['District'] = df_prop['District'].astype('category')
    df_prop['Market'] = df_prop['Market'].astype('category')
    df_prop['State_Code'] = df_prop['State'].cat.codes
    df_prop['District_Code'] = df_prop['District'].cat.codes
    df_prop['Market_Code'] = df_prop['Market'].cat.codes
    df_prop.head()

    state_dict = pd.Series(df_prop.State_Code.values, index=df_prop.State).to_dict()
    district_dict = pd.Series(df_prop.District_Code.values, index=df_prop.District).to_dict()
    market_dict = pd.Series(df_prop.Market_Code.values, index=df_prop.Market).to_dict()
    df_prop = df_prop.drop(['State', 'District', 'Market'], axis=1)
    df_prop.head()

    print(state_dict)

    test_number=48000
    train = df_prop[:-test_number]
    test = df_prop[-test_number:]

    prophet_model = Prophet(daily_seasonality = True)
    prophet_model.add_regressor('State_Code')
    prophet_model.add_regressor('District_Code')
    prophet_model.add_regressor('Market_Code')
    prophet_model.fit(train)

    df['Price'].hist()

    import pickle
    with open('model_train_test.pckl', 'wb') as fout:
        pickle.dump(prophet_model, fout)

    # # Model retrieval
    with open('model_train_test.pckl', 'rb') as fin:
        prophet_model = pickle.load(fin)

    test_wihtout_label = test.drop(['y'], axis=1)

    test_pred = prophet_model.predict(test_wihtout_label)

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    y_true = test['y']
    y_pred = test_pred['yhat']

    print( "Mean Absolute Error is", mean_absolute_error(y_true, y_pred)) 
    # print( "Max Error is", max_error(y_true, y_pred)) 
    # print( "Mean Squared Error is", mean_squared_error(y_true, y_pred)) 
    print( "R2 score is",r2_score(y_true, y_pred))

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mean_absolute_percentage_error(test['y'],test_pred['yhat'])

    fig1 = prophet_model.plot(test_pred)

    future = prophet_model.make_future_dataframe(periods=90)

    test_data = [['2022-02-14', '0', '11', '202']] 
    df_test = pd.DataFrame(test_data, columns = ['ds', 'State_Code', 'District_Code', 'Market_Code'])
    df_test.head()

    future.head()

    len(train)

    df_test.info()
    test_data = [['2015-02-06', '0', '11', '202']] 
    df_test = pd.DataFrame(test_data, columns = ['ds', 'State_Code', 'District_Code', 'Market_Code'])
    df_test.head()
    forecast_test = prophet_model.predict(df_test)
    forecast_test
    fig1 = prophet_model.plot(forecast_test)

    import datetime
    input_state = "Andhra Pradesh"
    input_district = "Anantapur"
    input_market = "Gooti"
    input_date = "15-03-15"

    State_Code = state_dict[input_state]
    District_Code = district_dict[input_district]
    Market_Code = market_dict[input_market]

    ds = datetime.datetime.strptime(input_date, "%d-%m-%y").strftime("%Y-%m-%d")
    ds = pd.to_datetime(ds)

    input_data = [[ds, State_Code, District_Code, Market_Code]]
    df_pred = pd.DataFrame(input_data, columns = ['ds', 'State_Code', 'District_Code', 'Market_Code'])

    #converting df columns to category dtype
    df_pred['State_Code'] = df_pred.State_Code.astype('category')
    df_pred['District_Code'] = df_pred.District_Code.astype('category')
    df_pred['Market_Code'] = df_pred.Market_Code.astype('category')

    pred_result = prophet_model.predict(df_pred)

    def get_week_prediction(input_state, input_district, input_market):
        import datetime
        from datetime import date
        from datetime import timedelta
        today = date.today()
        input_date = today + timedelta(days=1) #starting prediction week from tomorrow

        State_Code = state_dict[input_state]
        District_Code = district_dict[input_district]
        Market_Code = market_dict[input_market]

        input_data = []
        for i in range(7):
            input_data.append([input_date, State_Code, District_Code, Market_Code])
            input_date = input_date + timedelta(days=1) #date incremented by one day

        df_pred = pd.DataFrame(input_data, columns = ['ds', 'State_Code', 'District_Code', 'Market_Code'])
        input_data=[]

        #converting df columns to category dtype
        df_pred['State_Code'] = df_pred.State_Code.astype('category')
        df_pred['District_Code'] = df_pred.District_Code.astype('category')
        df_pred['Market_Code'] = df_pred.Market_Code.astype('category')

        return prophet_model.predict(df_pred)

    week_prediction_result = get_week_prediction("Andhra Pradesh","Anantapur", "Gooti")
    week_prediction_result

    df = pd.read_csv('/content/sample_data/Data_modified.csv')
    dummies_state = pd.get_dummies(df.State)
    dummies_day = pd.get_dummies(df.Day)
    dummies_month = pd.get_dummies(df.Month)
    dummies_year = pd.get_dummies(df.Year)
    dummies_district = pd.get_dummies(df.District)
    merged = pd.concat([df,dummies_day,dummies_month,dummies_year,dummies_state,dummies_district],axis='columns')
    final = merged.drop(['Day','Month','Year','State','District','Market'],axis='columns')
    X = final.drop('Price',axis='columns')
    y = final.Price
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df_ = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df_.head(25)
    accuracy = regressor.score(X_test,y_test)
    print(accuracy)

    df

    df[df['District']=='Ahmedabad' ]
    district_dict['Ahmedabad']

    df_prop

    test_data=df_prop[(df_prop['District_Code']==district_dict['Ahmedabad']) & (df_prop['State_Code']==state_dict['Gujarat'])]

    test_data=test_data.groupby(test_data.columns[0]).mean()

    test_data.columns

    x=np.array(test_data.index)

    y=np.array(test_data.y)

    plt.plot(x,y)
def mlmodel4():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os


    # In[2]:


    d=os.path.dirname(os.getcwd())
    d


    # In[3]:


    xn=os.path.join(d,"code\sih_2\sih")
    f=os.path.join(d,"code\sih_2\sih\States")
    e=os.path.join(f,"gujarat")
    q=os.path.join(e,"Amreli")
    os.chdir(q)


    # In[4]:


    dataset = pd.read_csv("feb.csv")
    x=dataset.iloc[:,:-1].values 
    y=dataset.iloc[:,-1].values 
    x,y,y_test=x[:-20],y[:-20],y[-20:]


    # In[5]:


    x,y


    # In[6]:


    size=y.size


    # In[7]:


    l=os.path.join(xn,"year")
    #j=os.path.join(l,"jan") 
    os.chdir(l)


    # In[8]:


    dataset2 = pd.read_csv("feb.csv")
    x2=dataset2.iloc[:,:-1].values
    #y2=dataset2.iloc[:,-1].values


    # In[9]:


    import math
    a=0
    yi=0
    for i in range(len(x)):
        a=a+x[i][2]
        yi=yi+x[i][3]   
    a/=size
    yi/=size


    # In[10]:


    for i in range(len(x2)):
        x2[i][2]=a
        x2[i][3]=yi


    # In[11]:


    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.compose import ColumnTransformer

    label_encoder_x_1 = LabelEncoder()
    x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
    transformer = ColumnTransformer(
        transformers=[
            ("OneHot",        
            OneHotEncoder(), 
            [0]              
            )
        ],
        remainder='passthrough' 
    )
    x = transformer.fit_transform(x.tolist())
    x = x.astype('float64')


    # In[12]:


    label_encoder_x_2 = LabelEncoder()
    x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
    transformer = ColumnTransformer(
        transformers=[
            ("OneHot",        
            OneHotEncoder(), 
            [1]              
            )
        ],
        remainder='passthrough' 
    )
    x = transformer.fit_transform(x.tolist())
    x = x.astype('float64')


    # In[13]:


    label_encoder_x_2 = LabelEncoder()
    x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
    transformer = ColumnTransformer(
        transformers=[
            ("OneHot",        
            OneHotEncoder(), 
            [0]              
            )
        ],
        remainder='passthrough' 
    )
    x2 = transformer.fit_transform(x2.tolist())
    x2 = x2.astype('float64')

    label_encoder_x_2 = LabelEncoder()
    x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
    transformer = ColumnTransformer(
        transformers=[
            ("OneHot",        
            OneHotEncoder(), 
            [1]             
            )
        ],
        remainder='passthrough' 
    )
    x2 = transformer.fit_transform(x2.tolist())
    x2 = x2.astype('float64')
    #x2=x2[:,1:]


    # In[14]:


    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=400,random_state=0)
    regressor.fit(x,y)

    y_pred=regressor.predict(x2)
    #plt.plot(y2,color='red',label='real')
    #plt.plot(y_pred,color='blue',label='pred')
    plt.title('Cotton price') 
    plt.xlabel('time')
    #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    xi = list(range(len(x)))
    plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
    plt.xticks(xi, x)
    plt.legend
    plt.show()


    # In[15]:


    y_test


    # In[16]:


    y_pred


    # In[17]:


    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import  mean_absolute_percentage_error
    from sklearn.metrics import r2_score

    y_true = y_test
    y_pred = y_pred
    print( "Max Error is", max_error(y_true, y_pred))
    print( "Mean Absolute Error is", mean_absolute_error(y_true, y_pred))
    print( "Mean Squared Error is", mean_squared_error(y_true, y_pred)) 
    print( "mean_absolute_percentage_error is", mean_absolute_percentage_error(y_true,y_pred)) 
