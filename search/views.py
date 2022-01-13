from django.shortcuts import render, redirect, get_object_or_404
from django.forms import model_to_dict
from django.views import View
from django.views.generic import (
    DetailView,
    ListView
)
from django.urls import reverse

from .forms import SearchTextForm

from .embed_search import search

# Create your views here.
class IndexView(View):
    template_name = "index.html"
    form_class = SearchTextForm

    def get(self, request, *args, **kwargs):
        context = {"form": self.form_class()}
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        form.is_valid()
        context = {"form": self.form_class(),
        "info": form.cleaned_data["searchText"],
        "results": search(form.cleaned_data["searchText"])}
        return render(request, self.template_name, context)


# class HomePageView(ListView):
#     template_name = "home/index.html"

#     def get_queryset(self):
#         disabled = self.kwargs.get("disabled")
#         if disabled is not None:
#             ids = set(Car.objects.values_list("carId", flat=True))
#             disabled = [int(i) for i in disabled.split()[:3]]
#             ids = ids.difference(disabled)
#             ids = random.sample(ids, 3)
#             queryset = Car.objects.filter(carId__in=ids)
#         else:
#             queryset = Car.objects.all()[:3]
#         return queryset

#     def get(self, request, *args, **kwargs):
#         self.kwargs["disabled"] = request.GET.get("seen")
#         return super().get(request, *args, **kwargs)