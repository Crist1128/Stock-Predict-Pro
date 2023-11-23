# models.py in admin_app

from django.db import models
from users_app.models import User  # 从users_app导入User模型

class Administrators(models.Model):
    admin_id = models.AutoField(primary_key=True)  # 管理员ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    role_id = models.ForeignKey('Roles', on_delete=models.CASCADE)  # 外键链接到Roles表
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间，日期时间型，自动添加

class Servers(models.Model):
    server_id = models.AutoField(primary_key=True)  # 服务器ID，自增主键
    server_name = models.CharField(max_length=255)  # 服务器名称，字符型，最大长度为255
    configuration = models.TextField()  # 配置，文本型
    load_balancing_status = models.CharField(max_length=20)  # 负载均衡状态，字符型，最大长度为20
    last_monitored_at = models.DateTimeField()  # 最后监测时间，日期时间型

class DataSources(models.Model):
    data_source_id = models.AutoField(primary_key=True)  # 数据源ID，自增主键
    data_source_name = models.CharField(max_length=255)  # 数据源名称，字符型，最大长度为255
    data_quality = models.CharField(max_length=20)  # 数据质量，字符型，最大长度为20
    data_integrity = models.CharField(max_length=20)  # 数据完整性，字符型，最大长度为20
    backup_status = models.CharField(max_length=20)  # 备份状态，字符型，最大长度为20

class Models(models.Model):
    model_id = models.AutoField(primary_key=True)  # 模型ID，自增主键
    model_name = models.CharField(max_length=255)  # 模型名称，字符型，最大长度为255
    data_source_id = models.ForeignKey(DataSources, on_delete=models.CASCADE)  # 外键链接到DataSources表
    training_status = models.CharField(max_length=20)  # 训练状态，字符型，最大长度为20
    model_performance = models.CharField(max_length=255)  # 模型性能，字符型，最大长度为255

class Logs(models.Model):
    log_id = models.AutoField(primary_key=True)  # 日志ID，自增主键
    log_type = models.CharField(max_length=20)  # 日志类型，字符型，最大长度为20
    log_content = models.TextField()  # 日志内容，文本型
    timestamp = models.DateTimeField(auto_now_add=True)  # 时间戳，日期时间型，自动添加

class SecurityEvents(models.Model):
    event_id = models.AutoField(primary_key=True)  # 事件ID，自增主键
    event_type = models.CharField(max_length=20)  # 事件类型，字符型，最大长度为20
    event_description = models.TextField()  # 事件描述，文本型
    timestamp = models.DateTimeField(auto_now_add=True)  # 时间戳，日期时间型，自动添加

class Roles(models.Model):
    role_id = models.AutoField(primary_key=True)  # 角色ID，自增主键
    role_name = models.CharField(max_length=255)  # 角色名称，字符型，最大长度为255

class UserPermissions(models.Model):
    record_id = models.AutoField(primary_key=True)  # 记录ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    role = models.ForeignKey(Roles, on_delete=models.CASCADE)  # 外键链接到Roles表

class Configurations(models.Model):
    config_name = models.CharField(max_length=255, primary_key=True)  # 配置项名称，字符型，最大长度为255，主键
    config_value = models.CharField(max_length=255)  # 配置值，字符型，最大长度为255

class UserFeedbacks(models.Model):
    feedback_id = models.AutoField(primary_key=True)  # 反馈ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    feedback_content = models.TextField()  # 反馈内容，文本型
    submission_time = models.DateTimeField(auto_now_add=True)  # 提交时间，日期时间型，自动添加
