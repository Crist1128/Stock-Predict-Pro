# models.py in users_app

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

# 自定义用户管理器
class CustomUserManager(BaseUserManager):
    def create_user(self, email, username, password=None, **extra_fields):
        """
        创建普通用户的方法。

        参数:
        - email: 用户的电子邮件地址
        - username: 用户名
        - password: 用户的密码
        - extra_fields: 额外的字段，可包含用户类型、电话号码等

        返回:
        - 新创建的用户实例
        """
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, password=None, **extra_fields):
        """
        创建超级用户（管理员）的方法。

        参数:
        - email: 用户的电子邮件地址
        - username: 用户名
        - password: 用户的密码
        - extra_fields: 额外的字段，可包含用户类型、电话号码等

        返回:
        - 新创建的超级用户实例
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        return self.create_user(email, username, password, **extra_fields)

# 用户模型
class User(AbstractBaseUser):
    USER_TYPES = (
        ('ordinary', '普通用户'),
        ('admin', '管理员'),
    )

    user_id = models.AutoField(primary_key=True)  # 用户ID，自增主键
    email = models.EmailField(unique=True)  # 电子邮件，唯一
    username = models.CharField(max_length=30, unique=True)  # 用户名，字符型，最大长度为30，唯一
    phone = models.CharField(max_length=15, blank=True, null=True)  # 手机号码，字符型，最大长度为15，可选
    ip_address = models.GenericIPAddressField(blank=True, null=True)  # IP地址，通用IP地址型，可选
    user_type = models.CharField(max_length=10, choices=USER_TYPES, default='ordinary')  # 用户类型，字符型，选项为普通用户或管理员，默认为普通用户
    verification_token = models.CharField(max_length=100, blank=True, null=True)  # 验证令牌，字符型，最大长度为100，可选
    reset_token = models.CharField(max_length=100, blank=True, null=True)  # 重置令牌，字符型，最大长度为100，可选
    token_expiration_time = models.DateTimeField(blank=True, null=True)  # 令牌过期时间，日期时间型，可选

    # 用户活动状态
    is_active = models.BooleanField(default=True)  # 用户活动状态，默认为活动
    # 用户是否是管理员
    is_staff = models.BooleanField(default=False)  # 用户是否是管理员，默认为非管理员

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.username

# 用户个人资料模型
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # 外键链接到User表
    notification_preferences = models.CharField(max_length=255)  # 通知偏好设置，字符型，最大长度为255

# 订阅和通知模型
class SubscriptionNotification(models.Model):
    subscription_id = models.AutoField(primary_key=True)  # 订阅ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    security_id = models.CharField(max_length=10)  # 股票/基金ID，字符型，最大长度为10
    subscription_status = models.CharField(max_length=20)  # 订阅状态，字符型，最大长度为20
    notification_preferences = models.CharField(max_length=255)  # 通知偏好设置，字符型，最大长度为255

# 密码重置请求模型
class PasswordResetRequest(models.Model):
    request_id = models.AutoField(primary_key=True)  # 请求ID，自增主键
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键链接到User表
    reset_token = models.CharField(max_length=100)  # 重置令牌，字符型，最大长度为100
    expiration_time = models.DateTimeField()  # 过期时间，日期时间型
