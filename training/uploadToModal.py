import io
import modal
import os

vol = modal.Volume.from_name("mistral_models")
localDataPath = "../data/confessions_Archive_100k_clean_TRAINING.jsonl"
app = modal.App("mistral-confessions")


@app.function()
def upload():
    with vol.batch_upload() as batch:
        batch.put_file(localDataPath, "/confessions_Archive_100k_clean_TRAINING.jsonl")



@app.function(volumes={"/data": vol})
def debug_paths():
    print("Current working directory:", os.getcwd())

    print("\nContents of /data:")
    for root, dirs, files in os.walk("/data"):
        print(root)
        for f in files:
            print("  ", f)



@app.local_entrypoint()
def main():
    debug_paths.remote()