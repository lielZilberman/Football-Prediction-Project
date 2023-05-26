from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from os import path
import json
import os
from django.conf import settings


# Create your views here.

def index(request):
    exportMetaDataFilePath = os.path.join(settings.BASE_DIR, 'exportedModels', 'metaData.json')
    # exportMetaDataFilePath = path.abspath(path.dirname(__name__)) + '/../exportedModels/metaData.json'
    print('/n')
    print(exportMetaDataFilePath)

    exportMetaData = {
      'home_teams': {},
      'away_teams': {},
    }
    if path.exists(exportMetaDataFilePath):
        exportMetaDataFile = open(exportMetaDataFilePath, 'r')
        exportMetaData = json.load(exportMetaDataFile)
        # print(exportMetaData)

    template = loader.get_template('web/index.html')
    context = {
        'home_teams': exportMetaData['home_teams'],
        'away_teams': exportMetaData['away_teams'],
    }
    return HttpResponse(template.render(context, request))