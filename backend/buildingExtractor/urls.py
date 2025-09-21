from django.urls import path
from . import views

urlpatterns = [
    path("buildings/", views.process_image, name="process_image"),
    path("new-buildings/", views.new_buildings_view, name="new_buildings_view"),
    path("buildings-in-a-year/", views.buildings_by_year, name="buildings_by_year"),
]