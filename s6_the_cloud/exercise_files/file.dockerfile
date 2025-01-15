FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY s6_the_cloud/exercise_files/main.py s6_the_cloud/exercise_files/main.py
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main.py"]
