FROM python:3.11-slim-buster

RUN apt-get update > /dev/null
# Ensure libgomp1 is installed - that will avoid memory allocation issues in the Thread-Local Storage (TLS) block
RUN apt-get -y install gcc python3-dev libpq-dev libgomp1 > /dev/null
RUN python3 -m pip install poetry

WORKDIR /usr/src/app

# Add LD_PRELOAD to fix TLS block memory allocation issues
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Add this line to set PYTHONPATH within the container
ENV PYTHONPATH="/usr/src/app/src"

# Set immutable environment variables for configuration and output paths
ENV PATH_TO_THE_CONFIGURATION_FILE=/usr/src/app/config_file.json
ENV PATH_TO_SAVE_THE_OUTCOMES=/usr/src/app/experiments

COPY pyproject.toml ./
COPY poetry.lock ./
COPY src src
COPY config_file.json ./

# create a folder to save the outcomes of the experiment
RUN mkdir -p /usr/src/app/experiments

RUN poetry install

CMD ["poetry", "run", "python", "-u", "src/database_generator/main.py"]