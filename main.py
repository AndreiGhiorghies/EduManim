import sys

from backend.agent.run import run_agent

if __name__ == "__main__":
    sys.exit(run_agent())



# for Docker:

# Run this only one time to build the Docker image. After that, you can run the container using the other command below.
# docker build -t devmaster-local -f Dockerfile .

# Run the Docker container with the following command:
# Windows(PowerShell): docker run -d --name edu-manim -v ${PWD}:/app devmaster-local
# Linux(Ubuntu): docker run -d --name edu-manim -v $(pwd):/app devmaster-local

# To access the running container, you can use the following command, where you can run the python scripts:
# docker exec -it edu-manim bash

# If you want to run directly the script main.py without entering the container, you can use the following command:
# docker exec -it edu-manim python main.py

# You can edit files locally, and just run the last command.

# For local model I used the model qwen2.5-3b-instruct-q4_k_m.gguf. You can download it by running the init.py script. It will download the model and place it in the Models folder. The model is about 2GB in size. Do not run in container, run it locally.

# For local model on gpu platform to work:
# pip uninstall -y llama-cpp-python --break-system-packages
# CMAKE_ARGS="-DLLAMA_HIPBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt --no-cache-dir --break-system-packages