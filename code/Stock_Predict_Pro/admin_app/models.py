# models.py in admin_app

from django.db import models
from users_app.models import User  # 导入User模型

class Administrator(models.Model):
    admin_id = models.AutoField(primary_key=True)  # 管理员ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    role_id = models.ForeignKey('Role', on_delete=models.CASCADE)  # 外键链接到Role表

class Server(models.Model):
    server_id = models.AutoField(primary_key=True)  # 服务器ID，自增主键
    server_name = models.CharField(max_length=255)  # 服务器名称，字符型，最大长度为255
    configuration = models.TextField()  # 配置信息，文本型
    load_balancing_status = models.CharField(max_length=20)  # 负载均衡状态，字符型，最大长度为20
    
class DataSource(models.Model):
    data_source_id = models.AutoField(primary_key=True)  # 数据源ID，自增主键
    data_source_name = models.CharField(max_length=255)  # 数据源名称，字符型，最大长度为255
    data_quality = models.CharField(max_length=20)  # 数据质量，字符型，最大长度为20
    data_integrity = models.CharField(max_length=20)  # 数据完整性，字符型，最大长度为20
    backup_status = models.CharField(max_length=20)  # 备份状态，字符型，最大长度为20

class Model(models.Model):
    model_id = models.AutoField(primary_key=True)  # 模型ID，自增主键
    model_name = models.CharField(max_length=255)  # 模型名称，字符型，最大长度为255
    data_source_id = models.ForeignKey(DataSource, on_delete=models.CASCADE)  # 外键链接到DataSource表
    training_status = models.CharField(max_length=20)  # 训练状态，字符型，最大长度为20
    model_performance = models.CharField(max_length=255)  # 模型性能，字符型，最大长度为255

class Log(models.Model):
    log_id = models.AutoField(primary_key=True)  # 日志ID，自增主键
    log_type = models.CharField(max_length=20)  # 日志类型，字符型，最大长度为20
    log_content = models.TextField()  # 日志内容，文本型

class SecurityEvent(models.Model):
    event_id = models.AutoField(primary_key=True)  # 安全事件ID，自增主键
    event_type = models.CharField(max_length=20)  # 事件类型，字符型，最大长度为20
    event_description = models.TextField()  # 事件描述，文本型

class Role(models.Model):
    role_id = models.AutoField(primary_key=True)  # 角色ID，自增主键
    role_name = models.CharField(max_length=255)  # 角色名称，字符型，最大长度为255

class UserPermission(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    role = models.ForeignKey(Role, on_delete=models.CASCADE)  # 外键链接到Role表

class Configuration(models.Model):
    config_name = models.CharField(max_length=255, primary_key=True)  # 配置名称，字符型，最大长度为255，主键
    config_value = models.CharField(max_length=255)  # 配置值，字符型，最大长度为255

class UserFeedback(models.Model):
    feedback_id = models.AutoField(primary_key=True)  # 用户反馈ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    feedback_content = models.TextField()  # 反馈内容，文本型
