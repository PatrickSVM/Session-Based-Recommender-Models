import os
import shutil
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY


class Gfile:
    def __init__(self, path, proj_name) -> None:
        self.storage_client = None
        self.bucket = None
        self.dir = os.path.dirname(path)

        if path.startswith("gs://"):
            bucket_name = self.dir.split("/", 3)[2]
            self.dir = self.dir.split("/", 3)[3]
            self.storage_client = storage.Client(proj_name)
            self.bucket = self.storage_client.bucket(bucket_name)

    def makedirs(self, path: str):
        if not self.bucket:
            os.makedirs(path, exist_ok=True)

    def fetch_file(self, file_name: str, tmp_dir: str):
        if not self.bucket:
            return os.path.join(self.dir, file_name)

        blob = self.bucket.blob(os.path.join(self.dir, file_name))
        path_to = os.path.join(tmp_dir, file_name)
        blob.download_to_filename(path_to, retry=DEFAULT_RETRY)
        return path_to

    def copy(self, path_from, path_to):
        if not self.bucket:
            os.makedirs(os.path.dirname(path_to), exist_ok=True)
            shutil.copy2(path_from, path_to)
        else:
            blob = self.bucket.blob(path_to)
            blob.upload_from_filename(path_from, retry=DEFAULT_RETRY)

    def open(self, file_name, mode):
        if not self.bucket:
            return open(os.path.join(self.dir, file_name), mode)
        else:
            blob = self.bucket.blob(os.path.join(self.dir, file_name))
            return blob.open(mode)

    def list_blobs(self, path_suffix="") -> list:
        if not self.bucket:
            return os.listdir(os.path.join(self.dir, path_suffix))
        else:
            blobs = self.bucket.list_blobs(
                prefix=os.path.join(self.dir, path_suffix),
                delimiter="/",
            )
            return [blob.name.removeprefix(f"{self.dir}/") for blob in blobs]

    def close(self):
        if self.storage_client:
            self.storage_client.close()
            self.bucket = None
            self.storage_client = None
