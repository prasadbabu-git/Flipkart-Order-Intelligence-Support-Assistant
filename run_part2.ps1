$ErrorActionPreference = 'Stop'
python -m pip install -r requirements.txt
python part2/train_classifier.py
python validate_project.py
