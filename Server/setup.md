# UI-TARS-7B Server Setup Guide

This guide outlines the steps to set up a server for running the UI-TARS-7B model for inference. For the purpose of this bachelor thesis, we will use Option 2: Host on a Self-Configured Server.

## Option 1: Hugging Face Inference Endpoints Cloud Deployment

This option leverages Hugging Face's managed infrastructure for easy deployment.

**Guidance:**

Follow the detailed instructions provided in the official UI-TARS `README_deploy.md` file on the project's GitHub repository.

[https://github.com/bytedance/UI-TARS/blob/main/README_deploy.md](https://github.com/bytedance/UI-TARS/blob/main/README_deploy.md)

## Option 2: Host on a Self-Configured Server

This option gives you more control over your environment, such as using a cloud VM or a local machine.

**Hardware Configuration:**

For the 7B model, an **RTX 6000 Ada or similar GPU** is recommended for sufficient performance.

**Environment Setup:**

This setup requires a CUDA-enabled environment. We recommend using the following RunPod environment as an example, but any similar CUDA-enabled setup should work:

* **RunPod PyTorch 2.4.0 Environment (Example):**
    * **Image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
    * **Includes:**
        * CUDA 12.4.1
        * PyTorch 2.4.0
        * Python 3.11

**Setup Steps:**

1.  **Install Hugging Face CLI tools:**
    Install the `huggingface_hub` package to download the model.

    ```bash
    pip install -U huggingface_hub
    ```

2.  **Download the 7B model:**
    Use the Hugging Face CLI to download the `bytedance-research/UI-TARS-7B-DPO` model. Choose a suitable local directory on your server.

    ```bash
    huggingface-cli download bytedance-research/UI-TARS-7B-DPO --local-dir /path/to/your/model
    ```

3.  **Create the `preprocessor_config.json` file:**
    Create this configuration file within the directory where you downloaded the model. Ensure the path matches your chosen model directory.

    ```bash
    cat > /path/to/your/model/preprocessor_config.json << 'EOL'
    {
      "do_normalize": true,
      "do_resize": true,
      "feature_extractor_type": "Qwen2VLImageProcessor",
      "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "processor_class": "Qwen2VLProcessor",
      "size": {
        "shortest_edge": 1080,
        "longest_edge": 1920
      },
      "do_center_crop": true
    }
    EOL
    ```

4.  **Install `vllm`:**
    Install the `vllm` library, which is used for efficient inference.

    ```bash
    pip install vllm==0.6.6
    ```

5.  **Start the server:**
    Run the `vllm` OpenAI-compatible API server. Point the `--model` argument to your chosen model directory and select a suitable `--port` for the server to listen on.

    ```bash
    python -m vllm.entrypoints.openai.api_server \
        --served-model-name ui-tars \
        --model /path/to/your/model \
        --trust-remote-code \
        --port 4000 \
        --limit-mm-per-prompt image=5
    ```

This will start an API server that you can then use to interact with the UI-TARS-7B model for inference.