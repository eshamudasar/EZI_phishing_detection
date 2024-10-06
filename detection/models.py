from django.db import models

# Create your models here.

class DatasetFile(models.Model):
    file = models.FileField(upload_to='dataset/')

    def __str__(self):
        return self.file.name