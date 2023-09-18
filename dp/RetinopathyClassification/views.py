from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np

def home(request):
    return render(request,'Home.html')

def result(request):
    if request.method == 'POST':
        rf=joblib.load('model_final.sav')
        ss=joblib.load('scaler.sav')
        input_values=[]
        diagnosis_options={0:"does not have diabetic retinopathy",1:"have diabetic retinopathy"}
    
        input_values.append(request.POST['age'])
        input_values.append(request.POST['systolic_bp'])
        input_values.append(request.POST['diastolic_bp'])
        input_values.append(request.POST['cholesterol'])

        input_array=np.array(input_values)
        input_array=input_array.reshape(1,-1)
        input_array=ss.transform(input_array)
        result=rf.predict(input_array)
        result_text=diagnosis_options[result[0]]
        print(result_text)
        a=rf.predict_proba(input_array)
        prob="{:.0%}". format(a[0,result[0]])
        return render(request,'results.html',{'result_text':result_text,'prob_text':prob})
    else:
        return render(request,'resultserror.html')




def about(request):
    return render(request,'about.html')
