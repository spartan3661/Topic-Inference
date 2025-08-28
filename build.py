
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from azure.storage.blob import BlobServiceClient
import os, shutil, tempfile
from azure.core.exceptions import ResourceExistsError
from dotenv import load_dotenv

def add_model_to_azure(conn_str, container, blob_name, model_dir):
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    service_client = BlobServiceClient.from_connection_string(conn_str)
    try:
        service_client.create_container(container)
    except ResourceExistsError:
        pass

    with tempfile.TemporaryDirectory() as tmpd:
        base_name = os.path.join(tmpd, "model_archive")
        archive_path = shutil.make_archive(base_name, "zip", root_dir=model_dir)

        client = service_client.get_blob_client(container, blob_name)
        with open(archive_path, "rb") as f:
            client.upload_blob(f, overwrite=True)
    print(f"Uploaded model archive to {container}/{blob_name}")

def train_model(container, in_blob, start=None, end=None, tz_name = "America/Chicago"):
    if not os.path.exists(in_blob):
        print("Data not found locally, downloading data...")
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob = client.get_blob_client(container, in_blob)

        with open(in_blob, "wb") as f:
            downloader = blob.download_blob()
            for chunk in downloader.chunks():
                f.write(chunk)


    df = pd.read_json(in_blob, lines=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time", "question"])
    df["question"] = (df["question"]
                      .str.lower()
                      .str.replace(r"https?://\S+"," ", regex=True)
                      .str.replace(r"\s+"," ", regex=True)
                      .str.strip()
                      )


    partial_df = df.sample(frac=1, random_state=42) # shuffles
    #partial_df[:3]

    print("Training new models...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(verbose=True, calculate_probabilities=False)

    embeddings = embed_model.encode(partial_df["question"].tolist(),
                                    batch_size=256,
                                    normalize_embeddings=True,
                                    show_progress_bar=True)
    topics, _ = topic_model.fit_transform(partial_df["question"].tolist(), embeddings=embeddings)

    print("Saving...")
    topic_model.save(f"{MODEL_DIR}/topic_model", save_embedding_model=True)

    print("Uploading...")
    add_model_to_azure(AZURE_CONN_STR, MODEL_CONTAINER, MODEL_BLOB_NAME, MODEL_DIR)



if __name__ == "__main__":
    load_dotenv()
    AZURE_CONN_STR = os.environ["AZURE_CSV_CONTAINER"]
    DATA_CONTAINER = "chat-logs"
    DATA_BLOB_NAME = "quora_test"

    SUMMARY_CONTAINER = "weekly-summaries"
    SUMMARY_BLOB_NAME = "topic_discovery.txt"

    MODEL_CONTAINER = "models"
    MODEL_BLOB_NAME = "topic_inference.zip"

    MODEL_DIR = "./topic_model_dir"
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)


    train_model(DATA_CONTAINER, DATA_BLOB_NAME)
