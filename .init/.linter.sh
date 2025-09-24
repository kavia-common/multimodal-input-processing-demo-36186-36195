#!/bin/bash
cd /home/kavia/workspace/code-generation/multimodal-input-processing-demo-36186-36195/multimodal_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

