"""Interactive studio and a stateless, validated inference endpoint."""
import json
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST


@ensure_csrf_cookie
@require_GET
def index(request):
    return render(request, "PigPredict_app/studio.html")


@require_GET
def legacy_entry(request):
    return redirect("index")


@require_POST
def predict(request):
    from .runtime import predict_recipe
    try:
        if len(request.body) > 16384:
            raise ValueError("Request is too large")
        payload = json.loads(request.body)
        result = predict_recipe(payload)
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(result)


@require_GET
def health(request):
    from .runtime import load_decoder
    metadata, _ = load_decoder()
    return JsonResponse({"status": "ready", "dyes": len(metadata["dyes"]),
                         "fabrics": metadata["fabrics"], "parameters": metadata["parameters"]})
