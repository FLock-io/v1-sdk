FROM python:3.11.3

WORKDIR /sdk

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 5000/tcp
