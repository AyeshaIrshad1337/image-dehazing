from django.contrib import admin
from django.urls import path
from dehazer import views
urlpatterns = [
    path('',views.index, name='home'),
     path('dehaze/', views.dehaze, name='dehaze'),
]
