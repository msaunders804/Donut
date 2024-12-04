'''from sentence_transformers import SentenceTransformer

# Download and save the model locally
SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models/all-MiniLM-L6-v2")
#SentenceTransformer("Phi-3.5-mini-instruct", cache_folder="./models/Phi-3.5")

from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name, cache_dir="./models/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/all-MiniLM-L6-v2")'''

from transformers import AutoModel, AutoTokenizer

# Define the model name
model_name = "sentence-transformers/distilbert/distilgpt2"

# Download the model and tokenizer, and save to local directory
model = AutoModel.from_pretrained(model_name, cache_dir="./models/distilbert/distilgpt2")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/distilbert/distilgpt2")

