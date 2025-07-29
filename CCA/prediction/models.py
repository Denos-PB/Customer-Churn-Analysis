from django.db import models

class PredictionHistory(models.Model):
    date_created = models.DateTimeField(auto_now_add=True)
    contract_type = models.CharField(max_length=20)
    monthly_charges = models.DecimalField(max_digits=10, decimal_places=2)
    total_charges = models.DecimalField(max_digits=10, decimal_places=2)
    tenure = models.IntegerField()
    prediction = models.BooleanField()

    class Meta:
        ordering = ['-date_created']