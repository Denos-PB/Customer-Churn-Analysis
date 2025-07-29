from django.shortcuts import render, redirect
from django.contrib import messages
from .ML_model import train_model, save_model, make_prediction
from .models import PredictionHistory
from joblib import load
import pandas as pd

def train_model_view(request):
    if request.method == 'POST':
        try:
            model,results = train_model()

            model_path=save_model(model)

            context = {
                'success' : True,
                'results' : results,
                'model_path' : model_path
            }

            messages.success(request, 'Model trained successfully!')

        except Exception as e:
            context = {
                'success' : False,
                'error': str(e)
            }

            messages.error(request, f'Error training model: {str(e)}')

        return render(request, 'prediction/training_results.html', context)

    return render(request, 'prediction/train.html')



def predict_view(request):
    if request.method == 'POST':
        data = {
            'contract': request.POST.get('contract'),
            'monthly_charges': float(request.POST.get('monthly_charges')),
            'total_charges': float(request.POST.get('total_charges')),
            'tenure': int(request.POST.get('tenure'))
        }

        prediction = make_prediction(data)

        PredictionHistory.objects.create(
            contract_type=data['contract'],
            monthly_charges=data['monthly_charges'],
            total_charges=data['total_charges'],
            tenure=data['tenure'],
            prediction=bool(prediction)
        )

        return render(request, 'prediction/result.html', {
            'prediction': prediction,
            **data
        })

    return render(request, 'prediction/predict.html')

def history_view(request):
    predictions = PredictionHistory.objects.all()
    return render(request, 'prediction/history.html', {
        'predictions': predictions
    })