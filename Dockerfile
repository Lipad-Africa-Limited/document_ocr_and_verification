# base image
FROM python:3.10

# -----------------------------------
# update image os
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# create required folder
RUN mkdir /app

# -----------------------------------
# if you plan to use a GPU, you should install the 'tensorflow-gpu' package
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org tensorflow-gpu
# ----------------------------------- 
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org 
# -----------------------------------
# install dependancies from source code (always up-to-date)
RUN pip install --upgrade pip
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e . --index-url https://pypi.org/simple
# -----------------------------------

# Set the working directory in the container
WORKDIR /app
# -----------------------------------
# create required folder 
COPY . .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# If you have additional packages 
# Install utility for process debug (DISABLE ON PROD)
RUN apt-get -y install procps
RUN apt-get -y install lsof
RUN apt-get -y install net-tools
RUN apt-get -y install systemctl
RUN apt-get -y install nano
RUN apt-get -y install socat
RUN apt-get -y install netcat-openbsd

# environment variables
ARG BUILD_ID
ENV BUILD_ID=${BUILD_ID}
ENV PYTHONUNBUFFERED=1
# -----------------------------------
# run the app (re-configure port if necessary)
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8016"]
EXPOSE 8016