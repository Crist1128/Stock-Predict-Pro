# Generated by Django 4.1 on 2023-11-28 08:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("users_app", "0001_initial"),
        ("stocks_app", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Index",
            fields=[
                (
                    "index_code",
                    models.CharField(max_length=10, primary_key=True, serialize=False),
                ),
                ("index_name", models.CharField(max_length=255)),
                ("market", models.CharField(max_length=20)),
                ("index_information", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="Price",
            fields=[
                ("record_id", models.AutoField(primary_key=True, serialize=False)),
                ("timestamp", models.DateTimeField()),
                ("open_price", models.DecimalField(decimal_places=2, max_digits=10)),
                ("high_price", models.DecimalField(decimal_places=2, max_digits=10)),
                ("low_price", models.DecimalField(decimal_places=2, max_digits=10)),
                ("close_price", models.DecimalField(decimal_places=2, max_digits=10)),
                ("volume", models.IntegerField()),
                (
                    "adjusted_close",
                    models.DecimalField(decimal_places=2, max_digits=10),
                ),
            ],
        ),
        migrations.CreateModel(
            name="UserIndexSubscription",
            fields=[
                (
                    "subscription_id",
                    models.AutoField(primary_key=True, serialize=False),
                ),
                ("subscription_status", models.CharField(max_length=20)),
                ("notification_preferences", models.CharField(max_length=255)),
                (
                    "index",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="stocks_app.index",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="users_app.user"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="UserPredictionRequest",
            fields=[
                ("request_id", models.AutoField(primary_key=True, serialize=False)),
                ("prediction_interval", models.CharField(max_length=2)),
                ("timestamp", models.DateTimeField()),
                ("prediction_result", models.TextField()),
            ],
        ),
        migrations.RenameModel(
            old_name="TechnicalIndicators",
            new_name="TechnicalIndicator",
        ),
        migrations.RenameModel(
            old_name="UserSubscriptions",
            new_name="UserSubscription",
        ),
        migrations.RemoveField(
            model_name="userpredictionrequests",
            name="stock_symbol",
        ),
        migrations.RemoveField(
            model_name="userpredictionrequests",
            name="user",
        ),
        migrations.RenameField(
            model_name="news",
            old_name="stock_symbol",
            new_name="stock",
        ),
        migrations.RenameField(
            model_name="technicalindicator",
            old_name="stock_symbol",
            new_name="stock",
        ),
        migrations.RenameField(
            model_name="usersubscription",
            old_name="stock_symbol",
            new_name="stock",
        ),
        migrations.RenameModel(
            old_name="Stocks",
            new_name="Stock",
        ),
        migrations.DeleteModel(
            name="Prices",
        ),
        migrations.DeleteModel(
            name="UserPredictionRequests",
        ),
        migrations.AddField(
            model_name="userpredictionrequest",
            name="stock",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="stocks_app.stock"
            ),
        ),
        migrations.AddField(
            model_name="userpredictionrequest",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="users_app.user"
            ),
        ),
        migrations.AddField(
            model_name="price",
            name="stock",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="stocks_app.stock"
            ),
        ),
    ]