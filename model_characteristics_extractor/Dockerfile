# 1. Start with the base image
FROM python:3.12

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
# This is a common practice to leverage Docker's build cache
COPY . .
# 4. Install dependencies
RUN pip install -r requirements.txt

# 5. Copy the rest of the application files


# 6. Define the command to run the application
CMD ["python", "service.py"]