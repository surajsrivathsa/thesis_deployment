FROM python:3.8-slim
WORKDIR /api
COPY ./requirements.txt /api/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

COPY . /api
WORKDIR "/api/fastapi_webserver"

CMD ["uvicorn", "search_main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


