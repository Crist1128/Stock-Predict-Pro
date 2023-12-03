from django.contrib import admin

# Register your models here.
# admin.py
from .models import HotStock
from .models import News

admin.site.register(HotStock)  # 热门股票

admin.site.register(News)  # 股票新闻
