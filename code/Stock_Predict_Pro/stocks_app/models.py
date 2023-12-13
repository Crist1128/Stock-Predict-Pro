# models.py in stocks_app

from django.db import models
from users_app.models import User  # 导入User模型


class Stock(models.Model):
    stock_symbol = models.CharField(max_length=10, primary_key=True)  # 股票代码，字符型，最大长度为10，主键
    company_name = models.CharField(max_length=255)  # 公司名称，字符型，最大长度为255
    market = models.CharField(max_length=20)  # 市场，字符型，最大长度为20
    company_profile = models.TextField(null=True, blank=True)  # 公司简介，文本型
    type = 'stock'  # 用来区分指数和股票

    def __str__(self):
        return self.company_name


class Price(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)  # 外键链接到Stock表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    open_price = models.DecimalField(max_digits=10, decimal_places=2)  # 开盘价，十进制型，最大位数为10，小数位数为2
    high_price = models.DecimalField(max_digits=10, decimal_places=2)  # 最高价，十进制型，最大位数为10，小数位数为2
    low_price = models.DecimalField(max_digits=10, decimal_places=2)  # 最低价，十进制型，最大位数为10，小数位数为2
    close_price = models.DecimalField(max_digits=10, decimal_places=2)  # 收盘价，十进制型，最大位数为10，小数位数为2
    volume = models.IntegerField()  # 成交量，整数型
    adjusted_close = models.DecimalField(max_digits=10, decimal_places=2)  # 调整后的收盘价，十进制型，最大位数为10，小数位数为2


class TechnicalIndicator(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)  # 外键链接到Stock表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    moving_average = models.DecimalField(max_digits=10, decimal_places=2)  # 移动平均线，十进制型，最大位数为10，小数位数为2
    relative_strength_index = models.DecimalField(max_digits=10, decimal_places=2)  # 相对强弱指数，十进制型，最大位数为10，小数位数为2
    stochastic_oscillator = models.DecimalField(max_digits=10, decimal_places=2)  # 随机指标，十进制型，最大位数为10，小数位数为2


class News(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)  # 外键链接到Stock表
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    news_title = models.CharField(max_length=255)  # 新闻标题，字符型，最大长度为255
    news_content = models.TextField()  # 新闻内容，文本型


class UserSubscription(models.Model):
    subscription_id = models.AutoField(primary_key=True)  # 订阅ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)  # 外键链接到Stock表
    subscription_status = models.CharField(max_length=20)  # 订阅状态，字符型，最大长度为20
    notification_preferences = models.CharField(max_length=255)  # 通知偏好设置，字符型，最大长度为255


class UserPredictionRequest(models.Model):
    request_id = models.AutoField(primary_key=True)  # 请求ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)  # 外键链接到Stock表
    prediction_interval = models.CharField(max_length=2)  # 预测时间间隔，字符型，最大长度为2
    timestamp = models.DateTimeField()  # 时间戳，日期时间型
    prediction_result = models.TextField()  # 预测结果，文本型


class Index(models.Model):
    index_code = models.CharField(max_length=10, primary_key=True)  # 指数代码，字符型，最大长度为10，主键
    index_name = models.CharField(max_length=255)  # 指数名称，字符型，最大长度为255
    market = models.CharField(max_length=20)  # 市场，字符型，最大长度为20
    index_information = models.TextField(null=True, blank=True)  # 指数信息，文本型
    type = 'index'  # 用来区分指数和股票


class UserIndexSubscription(models.Model):
    subscription_id = models.AutoField(primary_key=True)  # 订阅ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    index = models.ForeignKey(Index, on_delete=models.CASCADE)  # 外键链接到Index表
    subscription_status = models.CharField(max_length=20)  # 订阅状态，字符型，最大长度为20
    notification_preferences = models.CharField(max_length=255)  # 通知偏好设置，字符型，最大长度为255
