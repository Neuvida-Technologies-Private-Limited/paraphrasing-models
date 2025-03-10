FROM python:3.8-slim

# copy the requirements file into the image
COPY ./requirements.txt /application/requirements.txt

# switch working directory
WORKDIR /application

RUN pip install --upgrade pip

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /application

EXPOSE 8080

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]