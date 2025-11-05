import docker
import os
import sys

# --- Configuration ---
# The tag for your new image (e.g., 'my-app:latest')
IMAGE_TAG = 'my-python-app:v1'
# The path to the directory containing the Dockerfile (the build context)
# For this script, we assume the Dockerfile is in the same directory as the script.
BUILD_CONTEXT_PATH = os.path.dirname(os.path.abspath(__file__))
# The name of the Dockerfile (optional, defaults to 'Dockerfile')
DOCKERFILE_NAME = 'Dockerfile'

def build_docker_image(tag: str, path: str, dockerfile: str):
    """
    Builds a Docker image using the docker-py SDK.

    Args:
        tag (str): The tag to apply to the built image.
        path (str): The path to the build context directory.
        dockerfile (str): The name of the Dockerfile within the context path.
    """
    try:
        # 1. Initialize Docker Client
        # The client will automatically connect to the Docker daemon
        # using standard environment variables or default settings.
        client = docker.from_env()
        print(f"✅ Docker client connected successfully.")

        # 2. Start the Build Process
        print(f"\n🚀 Starting build for image: {tag} from context: {path}")
        print(f"   Using Dockerfile: {dockerfile}")

        # The client.images.build method handles the entire process.
        # It streams the logs as it builds.
        # 'path' is the build context directory.
        # 'tag' is the image tag.
        # 'dockerfile' specifies the Dockerfile name.
        # 'rm=True' removes intermediate containers after a successful build.
        image, logs = client.images.build(
            path=path,
            tag=tag,
            dockerfile=dockerfile,
            rm=True
        )

        # 3. Process and Print Build Logs (optional, but good for debugging)
        print("\n--- Build Log Output ---")
        for chunk in logs:
            if 'stream' in chunk:
                # Print each line of the build output
                sys.stdout.write(chunk['stream'])
        print("------------------------")


        # 4. Final Success Confirmation
        print(f"\n🎉 Successfully built Docker image: {image.tags[0]}")
        print(f"   Image ID: {image.short_id}")

    except docker.errors.APIError as e:
        print(f"\n❌ Docker API Error during build: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)

