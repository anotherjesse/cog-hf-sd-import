import io
import os
import shutil
import subprocess
import urllib
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


    def tar_dir(self, weights_dir, out_file):
        start = time.time()
        directory = Path(weights_dir)
        with tarfile.open(out_file, "w") as tar:
            for file_path in directory.rglob("*"):
                if file_path.is_dir():
                    continue
                print(file_path)
                tar.add(file_path, arcname=file_path.relative_to(directory))

        print("Made tar in {:.2f}s".format(time.time() - start))


    def predict(
        self,
        repo_id: str = Input(description="HF repo id: username/template", default="runwayml/stable-diffusion-v1-5"),
        revision: str = Input(description="HF repo revision", default="main")
    ) -> Path:

        weights_dir = "weights"
        if os.path.exists(weights_dir):
            shutil.rmtree(weights_dir)

        self.download_repo(repo_id, revision, weights_dir)

        out_file = "weights.tar"
        if os.path.exists(out_file):
            os.remove(out_file)

        self.tar_dir(weights_dir, out_file)

        return Path(out_file)
