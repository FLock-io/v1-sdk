FROM python:3.11.3

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# You should do any operations that require a network connection (e.g. downloading a HuggingFace model here)
# since the container won't have internet access.
# RUN python3 download_hf_model.py

EXPOSE 5000/tcp
CMD [ "python3", "example.py" ]
