#!/bin/bash

cd /home/Jose/Random/stonks/dashboard/financial-regime-analysis

uv run generate_metrics.py >> ./logs/metrics.log 2>&1