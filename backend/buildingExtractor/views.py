from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
from .get_buildings_polygon import run_building_extraction
from .compare_buildings import get_buildings_comparison
from .get_buildings_for_year import get_buildings_for_year
from django.conf import settings
import os

shapefile_path = os.path.join(settings.BASE_DIR, "buildingExtractor", "OAU_BOUNDARY.shp")


# Create your views here.
@csrf_exempt
def process_image(request):
    if request.method == "POST":
        print(tempfile.gettempdir())
        
        image_file = request.FILES["image"]


        # Save uploaded GeoTIFF temporarily
        temp_path = tempfile.mktemp(suffix=".tif")
        with open(temp_path, "wb+") as dest:
            for chunk in image_file.chunks():
                dest.write(chunk)

        # Run model → output polygons (GeoJSON)
        geojson_result = run_building_extraction(temp_path)

        os.remove(temp_path)

        return JsonResponse(geojson_result, safe=False)

    return JsonResponse({"error": "Invalid request"}, status=400)

@csrf_exempt
def new_buildings_view(request):
    year1 = int(request.GET.get("year1", None))
    year2 = int(request.GET.get("year2", None))

    print(f"Comparing buildings for years: {year1} and {year2}")

    geojson_data = get_buildings_comparison(year1, year2)
    return JsonResponse(geojson_data, safe=False)

@csrf_exempt
def buildings_by_year(request):
    year = int(request.GET.get("year", None))
    geojson = get_buildings_for_year(year)
    return JsonResponse(geojson, safe=False)