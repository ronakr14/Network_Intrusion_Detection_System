# Set base image (host OS)
FROM python

# By default, listen on port 5000
EXPOSE 5000

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .
COPY Random_Forest_Classifier_FS_DTC.pkl .
COPY templates/ ./templates/

# Install any dependencies
RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY app.py .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]