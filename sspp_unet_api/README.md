# SSPP-Unet

This directory contains all the code for our FastAPI backend, which provides four HTTP endpoint methods.

- `/`: Endpoint to health check, print `OK`.
- `/predict`: Endpoint to predict the masks for the uploaded brain tumor slices.
- `/upload`: Endpoint to upload a ZIP file containing brain tumor slices. Extracts files and stores them in a designated directory.
- `/download_sample`: Endpoint to provide a sample dataset for users to download.

## Usage

To test the FastAPI locally:
```sh
cd app/
python main.py
```

To create a docker image and test locally:
```sh
docker build -t <image_name> .
docker run -d <image_name>
```

If successful, you should see your FastAPI application running on `localhost:8000`. You can check `localhost:8000/docs` to test each endpoint method.
