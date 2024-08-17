from django.contrib import admin
from django.urls import path
from chat import views as views_chat

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/<str:ruc>', views_chat.read_jrxml_file, name='chat_bot')
]
