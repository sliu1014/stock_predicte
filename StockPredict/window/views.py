from django.shortcuts import render
from window.utils import save_img
# Create your views here.


def home(request):
    save_img()
    return render(request, "index.html")