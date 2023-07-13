FROM python:3.11.3

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY model/ model/

EXPOSE 5000/tcp
CMD [ "python", "./fedllm_example.py" ]
