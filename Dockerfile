FROM python:3.10-slim

WORKDIR /app

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git

COPY src/ .

CMD ["bash"]