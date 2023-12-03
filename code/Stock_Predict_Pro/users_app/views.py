from django.shortcuts import render
from django import forms
# Create your views here.


class search(forms.Form):
    search = forms.CharField(
        label='搜索栏',
        widget=forms.TextInput(attrs={'placeholder': '请输入想搜索的内容：'}),
        required=True
    )


def index_users(request):
    form = search()
    return render(request, 'users_app/index.html', {'form': form})
