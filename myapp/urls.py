from django.urls import path, re_path as url
from .views import *
from . import views

urlpatterns = [
    path('', views.home, name = "home"),
    path('portfolio', views.home, name = "home"),
    path('equity', views.equity, name = "equity"),
    path('risk', views.risk, name = "risk"),
    path('bond', views.bond, name = "bond"),
    # url(r'portfolio/', portfolio.as_view()),
    url(r'bondsSelected/', bondsSelected.as_view()),
    url(r'varData/', varData.as_view()),
]

# from django.urls import path
# from . import views
# from django.conf.urls import include, url
# from .views import *
# from rest_framework import routers
