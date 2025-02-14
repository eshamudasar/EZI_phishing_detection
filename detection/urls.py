from django.urls import path
from . import views  # Ensure you import your views
from django.conf import settings
from django.conf.urls.static import static
import os
urlpatterns = [
    path('', views.home, name='home'),  # Home view
    path('about/', views.about, name='about'),  # About view
    path('phishintro/', views.phishintro, name='phishintro'),  # Introduction to phishing
    path('login/', views.login, name='login'),  # Login view
    path('contact/', views.contact, name='contact'),  # Contact view
    path('download/<str:filename>/', views.download_file, name='download_file'),
    path('downloads/', views.downloads, name='downloads'),  # Downloads main page
    path('downloads/dataset.html', views.dataset_view, name='dataset_view'),  # Dataset download page 
    path('keyword_phishing/', views.keyword_phishing, name='keyword_phishing'),  # Keyword phishing view
    path('url_phishing/', views.url_phishing, name='url_phishing'),  # URL phishing view
    path('real_time_phishing/', views.real_time_phishing, name='real_time_phishing'),  # Real-time phishing view
    path('real_time_phishing/<str:filename>/', views.downloadfile, name='downloadfile'),
    
   
]+ static('/FYP/', document_root=settings.FYP_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
