from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from window.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", home),
]

urlpatterns += static(settings.STATIC_URL, default_root=settings.STATIC_ROOT)