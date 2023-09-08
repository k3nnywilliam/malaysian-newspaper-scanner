from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    html = "<html><body><h1>Think about it </h1>It is now.</body></html>"
    return HttpResponse(html)
