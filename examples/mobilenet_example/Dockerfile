FROM python:3.11.3

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000/tcp
CMD [ "python", "./mobilenet_example.py" ]
