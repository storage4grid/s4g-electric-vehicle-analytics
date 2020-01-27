"""
electric_vehicle_analytics URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from instructions import views


urlpatterns = [
    path('api/', include('instructions.urls')),
]
