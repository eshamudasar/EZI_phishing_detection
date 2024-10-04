from django.urls import path
from . import views  # Ensure you import your views

urlpatterns = [
    path('', views.home, name='home'),  # Home view
    path('about/', views.about, name='about'),  # About view
    path('login/', views.login, name='login'),
    path('downloads/', views.downloads, name='downloads'),  # Downloads view
    path('keyword_phishing/', views.keyword_phishing, name='keyword_phishing'),  # Keyword phishing view
    path('url_phishing/', views.url_phishing, name='url_phishing'),  # URL phishing view
    path('real_time_phishing/', views.real_time_phishing, name='real_time_phishing'),  # Add this line
]
