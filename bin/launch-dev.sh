#!/usr/bin/env bash
env FLASK_ENV=development FLASK_APP=./webservice/launcher.py flask run --host 0.0.0.0 --port 5000