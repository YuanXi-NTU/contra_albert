from django import forms


class SearchTextForm(forms.Form):
    searchText = forms.CharField(required=False)
