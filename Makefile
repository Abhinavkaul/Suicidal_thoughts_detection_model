PROJECT=Suicidal Thought Prediction

build-image:
	docker build -t test .

pytest: build-image
	docker run -it test /bin/bash -c "python -m pytest --cov=tests/ ./tests"

user-input: build-image
	docker run -it test /bin/bash -c "cd tests && python run.py"

docker-compose:
	docker-compose up
	docker-compose down