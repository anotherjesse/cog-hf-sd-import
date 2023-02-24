import io
import os
import shutil
import subprocess
import urllib
import zipfile
import tarfile
import time

from cog import BasePredictor, Input, Path
# from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline


class Predictor(BasePredictor):
    def download_repo(self, repo_id, revision, dest, cache_dir="diffusers-cache"):
        """Download the model weights from the given URL"""
        print("Downloading weights...")
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        pipe.save_pretrained(dest)


    def zip_dir(self, weights_dir, out_file):
        start = time.time()
        with zipfile.ZipFile(out_file, "w") as zip:
            directory = Path(weights_dir)
            print("adding to zip:")
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    print(file_path)
                    zip.write(
                        file_path,
                        arcname=file_path.relative_to(weights_dir)
                    )

        print("Made zip in {:.2f}s".format(time.time() - start))


    def predict(
        self,
        repo_id: str = Input(description="HF repo id: username/template", default="runwayml/stable-diffusion-v1-5"),
        revision: str = Input(description="HF repo revision", default="main")
    ) -> Path:

        weights_dir = "weights"
        if os.path.exists(weights_dir):
            shutil.rmtree(weights_dir)

        self.download_repo(repo_id, revision, weights_dir)

        out_file = "weights.zip"
        if os.path.exists(out_file):
            os.remove(out_file)

        self.zip_dir(weights_dir, out_file)

        return Path(out_file)
