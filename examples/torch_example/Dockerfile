FROM flock_base

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000/tcp
CMD [ "python", "./torch_example.py" ]
