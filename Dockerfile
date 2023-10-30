FROM python:3.11 as base

ARG PACKAGE_NAME="shopper"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY . /app/${PACKAGE_NAME}

ENV PACKAGE_NAME=$PACKAGE_NAME

# Run the start script
ENTRYPOINT /app/${PACKAGE_NAME}/scripts/start.sh


