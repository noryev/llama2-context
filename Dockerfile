FROM pytorch/pytorch:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install transformers \
    accelerate \
    bitsandbytes \
    sentencepiece

# Create model directory
RUN mkdir -p /app/model

# Create download script
RUN echo "import os\n\
from transformers import AutoTokenizer, AutoModelForCausalLM\n\
import torch\n\
import logging\n\
\n\
logging.basicConfig(level=logging.INFO)\n\
\n\
def download_model():\n\
    try:\n\
        model_id = \"facebook/opt-2.7b\"  # Using larger model\n\
        save_directory = \"/app/model\"\n\
        \n\
        logging.info(f\"Downloading model {model_id}...\")\n\
        model = AutoModelForCausalLM.from_pretrained(\n\
            model_id,\n\
            device_map=\"auto\",\n\
            torch_dtype=torch.float16,\n\
            low_cpu_mem_usage=True\n\
        )\n\
        model.save_pretrained(save_directory)\n\
        \n\
        logging.info(\"Downloading tokenizer...\")\n\
        tokenizer = AutoTokenizer.from_pretrained(model_id)\n\
        tokenizer.save_pretrained(save_directory)\n\
        \n\
        logging.info(\"Model and tokenizer downloaded successfully!\")\n\
        \n\
    except Exception as e:\n\
        logging.error(f\"Error downloading model: {str(e)}\")\n\
        raise\n\
\n\
if __name__ == \"__main__\":\n\
    download_model()" > download_model.py

# Download the model during build
RUN python download_model.py

# Copy the inference script
COPY run_llama.py /app/
RUN chmod +x /app/run_llama.py

ENTRYPOINT ["python", "/app/run_llama.py"]
