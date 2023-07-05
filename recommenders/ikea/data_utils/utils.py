import os
import gzip
import json
import wandb
from google.cloud import storage


def download_file_from_gcp(proj_name, file_adress, target_dir, target_name):
    """
    Function to download file from gct to local dir.
    """
    storage_client = storage.Client(project=proj_name)

    # If directory not existing, create
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file = open(os.path.join(target_dir, target_name), "wb")
    storage_client.download_blob_to_file(file_adress, file)
    file.close()

    print(f"File {target_name} successfully downloaded.")


def upload_file_from_local(proj_name, bucket_name, file_path, file_name):
    """
    Function to uplaod local file to GCP bucket.
    """
    storage_client = storage.Client(project=proj_name)
    bucket = storage_client.bucket(bucket_name)
    new_file = bucket.blob(file_name)

    new_file.upload_from_filename(file_path)

    print(f"File {file_path} successfully uploaded.")


def upload_csv_from_memory(df, proj_name, bucket_name, file_path, file_name):
    """
    Function to uplaod pandas df to csv file in GCP bucket.
    """
    if ".csv" not in file_name:
        file_name = file_name + ".csv"
    storage_client = storage.Client(project=proj_name)
    bucket = storage_client.bucket(bucket_name)

    # Convert to csv-string on memory (not on disk)
    csv_string = df.to_csv(index=False)

    # Create file and upload
    blob = bucket.blob(os.path.join(file_path, file_name))
    blob.upload_from_string(csv_string)

    print(f"CSV {os.path.join(file_path, file_name)} successfully uploaded.")


def save_txt_to_gcp(proj_name, bucket_name, file_name, content):
    client = storage.Client(project=proj_name)
    bucket = client.bucket(bucket_name)
    new_file = bucket.blob(file_name)

    with new_file.open("w") as f:
        f.write(content)

    print("Done writing file.")

    from google.cloud import storage


def upload_directory_to_gcp(directory_path, proj_name, bucket_name, subdir):
    # Create a client object for interacting with the Google Cloud Storage API
    client = storage.Client(project=proj_name)

    # Get the bucket object
    bucket = client.get_bucket(bucket_name)

    # Loop through each file in the directory and upload it to the bucket
    for filename in os.listdir(directory_path):
        # Create a blob object for the file
        blob = bucket.blob(os.path.join(subdir, filename))

        # Upload the file to the bucket
        blob.upload_from_filename(os.path.join(directory_path, filename))

    print(
        f"All files in directory {directory_path} were uploaded to bucket {bucket_name}."
    )


def load_json_to_list(file_name, gfile_obj):
    """
    Takes GFile object and file name pointing to gzip of json
    dump, opens it and saves it in list of dicts.
    """
    with gzip.open(
        gfile_obj.open(file_name=file_name, mode="rb"), mode="rt", encoding="u8"
    ) as file:
        content = []
        for line in file:
            json_line = json.loads(line)
            content.append(json_line)

    return content


def download_wandb_history(location, exp_class):
    """
    Download all experiment data from wandb for current
    run to local csv file in location directory.
    """
    api = wandb.Api()
    run = api.run(f"patricksvm/{exp_class}/{wandb.run._run_id}")
    data = run.history()

    # Save to csv
    data.to_csv(f"{location}/history.csv", header=True)

    print("Successfully downloaded wandb run data.")
