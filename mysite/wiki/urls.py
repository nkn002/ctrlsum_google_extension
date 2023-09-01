from django.urls import path
from django.urls import re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^get_wiki_summary/$', views.get_wiki_summary, name='get_wiki_summary'),
]
