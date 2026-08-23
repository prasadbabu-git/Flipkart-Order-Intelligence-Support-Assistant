setup:
	python -m pip install -r requirements.txt

part1:
	python generate_orders.py
	python part1/train_and_evaluate.py

part2:
	python part2/train_classifier.py

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run app/dashboard.py

test:
	pytest -q

validate:
	python scripts/validate_repo.py
