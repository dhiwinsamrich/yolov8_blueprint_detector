# Deploying to Render

This guide provides step-by-step instructions for deploying the Blueprint Detection API to Render.

## Prerequisites

1.  **A Render Account**: Sign up at [render.com](https://render.com).
2.  **Git Repository**: Your project code, including the `Dockerfile`, `app.py`, `requirements.txt`, and the `models/` directory (with your trained model), must be hosted on a Git provider supported by Render (GitHub, GitLab, Bitbucket).
3.  **Trained Model**: Ensure your primary trained model (e.g., `blueprint_detector.pt` or `best.pt`) is in the `models/` directory and committed to your Git repository.

## Deployment Steps

1.  **Create a New Web Service on Render**:
    *   Log in to your Render dashboard.
    *   Click the "New +" button and select "Web Service".

2.  **Connect Your Git Repository**:
    *   Choose your Git provider and select the repository containing your project.
    *   Render will automatically detect your `Dockerfile` if it's in the root of your repository.

3.  **Configure the Service**:
    *   **Name**: Give your service a unique name (e.g., `blueprint-detection-api`).
    *   **Region**: Choose a region closest to your users.
    *   **Branch**: Select the Git branch you want to deploy (e.g., `main` or `master`).
    *   **Root Directory**: Leave this blank if your `Dockerfile` is in the root of the repository. If it's in a subdirectory, specify the path.
    *   **Environment**: Select "Docker". Render should automatically detect your `Dockerfile`.
    *   **Instance Type**: Choose an appropriate instance type based on your expected load and budget (e.g., "Starter" or a paid plan for more resources).

4.  **Port Configuration**:
    *   Render automatically detects the port exposed in your `Dockerfile` (which is `8000` in this project) and uses the `PORT` environment variable internally. Your current `Dockerfile` and `app.py` (via Uvicorn's default behavior when `--port` is specified) are set up to listen on port `8000`, which is fine.

5.  **Environment Variables (Optional but Recommended)**:
    *   While your `Dockerfile` hardcodes the port for Uvicorn, Render services are typically designed to respect a `PORT` environment variable set by Render itself. Your current setup should work, but for future flexibility or if you encounter issues, you might consider modifying your `Dockerfile`'s `CMD` to use an environment variable for the port if Render requires it for certain instance types or configurations. However, for most Docker deployments, exposing the port and having Uvicorn listen on it is sufficient.
    *   If you have other configurations (e.g., API keys for other services, though not present in this project), add them under the "Environment" section.

6.  **Build and Deploy**:
    *   Click "Create Web Service".
    *   Render will pull your code, build the Docker image, and deploy your application.
    *   You can monitor the build and deployment logs in the Render dashboard.

7.  **Access Your Deployed API**:
    *   Once the deployment is successful, Render will provide you with a public URL (e.g., `https://your-service-name.onrender.com`).
    *   Your API will be available at this URL. For example, the health check endpoint would be `https://your-service-name.onrender.com/health` and the API documentation at `https://your-service-name.onrender.com/docs`.

## Auto-Deploy (Optional)

Render can automatically redeploy your application whenever you push changes to the configured Git branch. This is usually enabled by default.

## Troubleshooting

*   **Build Failures**: Check the build logs on Render for errors. Common issues include missing dependencies in `requirements.txt`, incorrect `Dockerfile` instructions, or problems accessing the model file.
*   **Application Errors**: Check the runtime logs for your service on Render. Ensure your `app.py` is starting correctly and that the model is loading.
*   **Model Not Found**: Double-check that your model file is correctly placed in the `models/` directory and that this directory is included in your Git repository and copied into the Docker image.

By following these steps, you should be able to successfully deploy your Blueprint Detection API to Render.