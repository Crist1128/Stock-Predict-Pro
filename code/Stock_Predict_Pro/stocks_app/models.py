# models.py in stocks_app

from django.db import models
from users_app.models import User  # 从users_app导入User模型


class Stocks(models.Model):
    stock_symbol = models.CharField(max_length=10, primary_key=True)  # 股票代码，字符型，最大长度为10，主键
    company_name = models.CharField(max_length=255)  # 公司名称，字符型，最大长度为255
    industry_classification = models.CharField(max_length=255)  # 行业分类，字符型，最大长度为255
    market = models.CharField(max_length=20)  # 市场，字符型，最大长度为20
    company_profile = models.TextField()  # 公司简介，文本型

    def __str__(self):
        return self.stock_symbol


class Prices(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock_symbol = models.ForeignKey(Stocks, on_delete=models.CASCADE)  # 外键链接到Stocks表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    open_price = models.DecimalField(max_digits=10, decimal_places=2)  # 开盘价，十进制数，最大位数为10，小数位数为2
    high_price = models.DecimalField(max_digits=10, decimal_places=2)  # 最高价，十进制数，最大位数为10，小数位数为2
    low_price = models.DecimalField(max_digits=10, decimal_places=2)  # 最低价，十进制数，最大位数为10，小数位数为2
    close_price = models.DecimalField(max_digits=10, decimal_places=2)  # 收盘价，十进制数，最大位数为10，小数位数为2
    volume = models.PositiveIntegerField()  # 成交量，正整数
    adjusted_close = models.DecimalField(max_digits=10, decimal_places=2)  # 调整后的收盘价，十进制数，最大位数为10，小数位数为2


class TechnicalIndicators(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock_symbol = models.ForeignKey(Stocks, on_delete=models.CASCADE)  # 外键链接到Stocks表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    moving_average = models.DecimalField(max_digits=10, decimal_places=2)  # 移动平均线，十进制数，最大位数为10，小数位数为2
    relative_strength_index = models.DecimalField(max_digits=10, decimal_places=2)  # 相对强弱指数，十进制数，最大位数为10，小数位数为2
    stochastic_oscillator = models.DecimalField(max_digits=10, decimal_places=2)  # 随机指标，十进制数，最大位数为10，小数位数为2


class News(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock_symbol = models.ForeignKey(Stocks, on_delete=models.CASCADE)  # 外键链接到Stocks表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    news_title = models.CharField(max_length=255)  # 新闻标题，字符型，最大长度为255
    news_content = models.TextField()  # 新闻内容，文本型
    objects = models.Manager()


class UserSubscriptions(models.Model):
    subscription_id = models.AutoField(primary_key=True)  # 订阅ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    stock_symbol = models.ForeignKey(Stocks, on_delete=models.CASCADE)  # 外键链接到Stocks表
    subscription_status = models.CharField(max_length=20)  # 订阅状态，字符型，最大长度为20
    notification_preferences = models.CharField(max_length=255)  # 通知偏好设置，字符型，最大长度为255


class UserPredictionRequests(models.Model):
    request_id = models.AutoField(primary_key=True)  # 请求ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    stock_symbol = models.ForeignKey(Stocks, on_delete=models.CASCADE)  # 外键链接到Stocks表
    prediction_interval = models.CharField(max_length=1, choices=[('M', '分钟'), ('H', '小时')])  # 预测时间间隔，字符型，选项为分钟或小时
    request_timestamp = models.DateTimeField()  # 请求时间戳，日期时间型
    prediction_result = models.CharField(max_length=255)  # 预测结果，字符型，最大长度为255


class HotStock(models.Model):
    stock_symbol = models.CharField(max_length=10)
    company_name = models.CharField(max_length=100)
    latest_close_price = models.FloatField()
    change_amount = models.FloatField()
    change_percentage = models.FloatField()
    objects = models.Manager()
