import base64
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from os import PathLike
from pathlib import Path

import replicate
import requests
from PIL import Image


def get_replicate_key(api_key_path: Path | str) -> str:
    api_key_path = os.path.join(api_key_path, "replicate.txt")
    # Lire la clé d'API depuis le fichier
    api_key = ""
    if os.path.exists(api_key_path):
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
    return api_key


NEG_PROMPT_GENERIC = "deformed, bad anatomy, disfigured, (poorly draw face:1.4), mutation, mutated, disconnected limb, malformed hands, out of frame, extra limbs, extra legs, extra arms, bad art"


class ImageGenerator_replicate:
    NEG_PROMPT = "deformed, bad anatomy, disfigured, (poorly draw face:1.4), mutation, mutated, disconnected limb, malformed hands, ugly, blur, blurry, fake, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, body out of frame, bad art, bad anatomy, ugly, poorly drawn, text, watermark, signature, logo, split image, copyright, desaturated"

    def __init__(self, api_token, max_workers=10):
        self.api_token = api_token
        #replicate.api_token = api_token
        self.api = replicate.Client(api_token=get_replicate_key('/home/yopla/Documents/keys/'))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.base_model = "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f"
        #self.base_model = "lucataco/dreamshaper-xl-turbo:0a1710e0187b01a255302738ca0158ff02a22f4638679533e111082f9dd1b615"
        #self.base_model = "shefa/turbo-enigma:d782a412563ca745fddce97c26ff3e72c551deba88a835188374a8a3ab9b43cc"
        #self.base_model = "charlesmccarthy/animagine-xl:db29f76d40ecf86335295ca5b24ed95e6b1eca4e29239c47cfefa68f408cbf5e"
        self.ghibli_model = "grabielairu/ghibli:4b82bb7dbb3b153882a0c34d7f2cbc4f7012ea7eaddb4f65c257a3403c9b3253"
        # ghibli format : input={
        #         "width": 1024,
        #         "height": 1024,
        #         "prompt": "full body shot, anime, emotional, dynamic, distortion for emotional effect, vibrant, A beautiful 21yo girl, blond twin braids, blue eyes, small breasts, on the forest, long shot, detailed skin, detailed eyes, ultra-detailed face, in the style of Milo Manara, highly detailed, sharp attention to detail, extremely detailed, dynamic composition\n",
        #         "refine": "expert_ensemble_refiner",
        #         "scheduler": "K_EULER",
        #         "lora_scale": 0.6,
        #         "num_outputs": 4,
        #         "guidance_scale": 7.5,
        #         "apply_watermark": True,
        #         "high_noise_frac": 0.8,
        #         "negative_prompt": "bad anatomy, bad hands, missing arms,  extra hands, extra fingers, bad fingers, extra legs, missing legs, poorly drawn face, fused face, worst feet, extra feet, fused feet, fused thigh, extra thigh, worst thigh, missing fingers, long fingers, extra eyes, huge eyes, amputation",
        #         "prompt_strength": 0.8,
        #         "num_inference_steps": 50
        #     }
        self.base_path = "model_"
        self.consistent_charactere_model = "fofr/consistent-character:9c77a3c2f884193fcee4d89645f02a0b9def9434f9e03cb98460456b831c8772"
        self.consistent_charactere_path = "_consistent_"
        #self.model_extract_person = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"
        self.model_extract_person = "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1"
        self.extract_person_path = "_character_"
        self_model_pose = "renyurui/controllable-person-synthesis:f2a5c4525dcd2868f7db9013e0ad82f82992ab5ce7418d6cc414e0ceca5861ec"

        self.bd_model = "hvision-nku/storydiffusion:39c85f153f00e4e9328cb3035b94559a8ec66170eb4c0618c07b16528bf20ac2"
        self.bd_path = "_bd_"
        self.bd_style_name = ["(No style)", "Japanese Anime", "Cinematic", "Disney Charactor", "Photographic",
                              "Comic book", "Line art"]
        self.bd_sd_model = ["Unstable", "RealVision"]
        self.bd_input = {
            "num_ids": 3,
            "sd_model": "Unstable",
            "num_steps": 25,
            "style_name": "Japanese Anime",
            "comic_style": "Classic Comic Style",
            "image_width": 512,
            "image_height": 512,
            "sa32_setting": 0.5,
            "sa64_setting": 0.5,
            "output_format": "webp",
            "guidance_scale": 5,
            "output_quality": 80,
            "negative_prompt": "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            "comic_description": """at school, dreaming about love
sitting alone on a park bench, aroused.
reading a book a hand touching her breast.
A man approaches, looking her. 
look around in the park. # She looks around and enjoys the beauty of nature, happy.
leaf falls from the tree, landing on the sketchbook.
picks up the leaf, examining its details closely.
The man appear.
She is very happy to see the man again
She stay here""",
            "style_strength_ratio": 20,
            "character_description": "a girl img",
            "disable_safety_checker": True
        }

        self.swap_model = "omniedgeio/face-swap:c2d783366e8d32e6e82c40682fab6b4c23b9c6eff2692c0cf7585fc16c238cfe"
        self.swap_path = "_swaped_"

    def __enter__(self):
        # Called when the class is instantiated in a 'with' block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Called when the 'with' block is exited
        self.executor.shutdown(wait=True)

    def generate_image(self, prompt, negative_prompt=None, model_id=None, lora_model=None, num_outputs=1,
                       num_inference_steps=20, guidance_scale=9, width=512, height=512, seed=None, nsfw=False,
                       output_file: PathLike = None, image_face=""):
        output_folder = os.path.dirname(output_file)
        base_name = os.path.basename(output_file)
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
        """
        [1] Subject, [2] Detailed Imagery, [3] Environment Description, [4] Mood/Atmosphere Description, [5] Style, [6] Style Execution
        """
        #print("Generate...")
        input_params = {"prompt": prompt, "num_outputs": num_outputs, "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale, "width": width, "height": height,
                        "sampler_index": "DPM + + 2M Karras", 'restore_face': True, 'disable_safety_checker': nsfw,
                        "negative_prompt": negative_prompt if negative_prompt else NEG_PROMPT_GENERIC,
                        "scheduler": "K_EULER"}

        if seed:
            input_params["seed"] = seed

        if model_id is None:
            model_id = self.base_model
        if "shefa/turbo-enigma" in model_id:
            if image_face:
                file = open(image_face, "rb")
                input_params["faceswap_fast"] = False
                input_params["faceswap_slow"] = True
                input_params["image"] = file
                input_params["guidance_scale"] = 1.77
                input_params[
                    "embeddings"] = "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"
                input_params["num_inference_steps"] = 7
                input_params["num_refine_steps"] = 3

        if "sdxl-lightning-4step" in model_id:
            #print("Model with 4 steps...")
            input_params["num_inference_steps"] = 4
            input_params["guidance_scale"] = 0

        if lora_model:
            input_params["lora_model"] = lora_model

        future = self.executor.submit(self.api.run, model_id, input=input_params)
        output_urls = future.result()
        output = []
        if "shefa/turbo-enigma" in model_id:
            response = requests.get(output_urls)
            #print(output_urls)
            file = os.path.join(output_folder, self.base_path + f"_0")
            with open(file, "wb") as f:
                output.append(file)
                f.write(response.content)
            return output
        for i, url in enumerate(output_urls):
            response = requests.get(url)
            #print(url)
            file = os.path.join(output_folder, base_name + f"_{i}.png")
            with open(file, "wb") as f:
                output.append(file)
                f.write(response.content)

        return output

    @staticmethod
    def base64_to_image(image_str):
        """Convert base64 encoded string to an image."""
        image = Image.open(BytesIO(base64.b64decode(image_str)))
        return image

    def generate_pose_character(self, desired_pose, image, output_file):
        file = open(image, "rb")
        input = {
            "seed": "729973005",
            "desired_pose": desired_pose,
            "reference_image": file
        }


        future = self.executor.submit(self.api.run, self.model_pose, input=input)
        output_urls = future.result()
        #print(output_urls)
        """
        with open(file, "wb") as f:
            f.write(response.content)
            output.append(file)
        """
        return output_urls

    def generate_consistent_character(self, prompt, base_subject, num_outputs=5, output_file="", nsfw=True,
                                      neg_prompt=""):
        output_folder = os.path.dirname(output_file)
        if output_folder and not os.path.exists(output_folder):
            os.mkdir(output_folder)

        image = open(base_subject, "rb")
        input_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "subject": image,
            "number_of_outputs": num_outputs,
            "output_quality": 100,
            "output_format": "png",
            'disable_safety_checker': nsfw
        }

        future = self.executor.submit(self.api.run, self.consistent_charactere_model, input=input_params)
        output_urls = future.result()
        image.close()  # Assurez-vous de fermer l'image après utilisation

        output = []
        for i, url in enumerate(output_urls):
            response = requests.get(url)
            file = output_file.replace(".png", f"_{i}.png")
            with open(file, "wb") as f:
                f.write(response.content)
            output.append(file)
        return output


    def _generate_consistent_character(self, prompt, base_subject, num_outputs=5, output_file="", nsfw=True, neg_prompt=""):
        output_folder = os.path.dirname(output_file)
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        #print("Consistent...")
        #sujet est une image
        image = open(base_subject, "rb")
        input_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "subject": image,
            "number_of_outputs": num_outputs,
            'disable_safety_checker': nsfw
        }

        future = self.executor.submit(self.api.run, self.consistent_charactere_model, input=input_params)
        output_urls = future.result()
        output = []

        for i, url in enumerate(output_urls):
            #print(url)
            response = requests.get(url)
            file = output_file.replace(".png", f"_{i}.png")

            with open(file, "wb") as f:
                f.write(response.content)
                output.append(file)
        return output

    def extract_personnage(self, input_image_path, output_file=""):

        output_folder = os.path.dirname(output_file)
        base_name = os.path.basename(output_file)

        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        #print(f"Extract {input_image_path}")
        image = open(input_image_path, "rb")
        input_params = {
            "image": image
        }

        future = self.executor.submit(self.api.run, self.model_extract_person, input=input_params)
        output_url = future.result()
        #print(output_url)
        output = []

        response = requests.get(output_url)

        with open(output_file, "wb") as f:
            f.write(response.content)
            output.append(output_file)
        #print(f"extract personnage files : {output}")
        return output

    def create_some_character_sprite(self, prompt, neg_prompt=NEG_PROMPT, model_id=None, lora_model=None, num_outputs=1,
                                     num_inference_steps=20, guidance_scale=9, width=512, height=512, seed=None,
                                     nsfw=False, output_folder="", nb_sprite=3):

        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        output_file = self.generate_image(prompt, neg_prompt, model_id, lora_model, num_outputs, num_inference_steps,
                                          guidance_scale, width, height, seed, nsfw, output_folder)
        output_file = self.generate_consistent_character(prompt, base_subject=output_file[0], num_outputs=nb_sprite,
                                                         output_folder=output_folder)
        for file in output_file:
            out = self.extract_personnage(file, output_folder)

    def generate_bd(self, input_image_path, output_folder=""):

        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
        print("bd...")
        image = open(input_image_path, "rb")

        input_params = self.bd_input
        input_params["ref_image"] = image
        future = self.executor.submit(self.api.run, self.bd_model, input=input_params)
        output_url = future.result()
        output_urls = output_url["individual_images"]
        print(output_url)
        output = []
        for i, url in enumerate(output_urls):
            print(url)
            response = requests.get(url)
            file = os.path.join(output_folder,
                                os.path.basename(input_image_path) + self.bd_path + str(i) + ".png")
            with open(file, "wb") as f:
                f.write(response.content)
                output.append(file)

        """
        https://replicate.com/hvision-nku/storydiffusion
        OUTPUT FORMAT
        {
  "type": "object",
  "title": "ModelOutput",
  "required": [
    "comic",
    "individual_images"
  ],
  "properties": {
    "comic": {
      "type": "string",
      "title": "Comic",
      "format": "uri"
    },
    "individual_images": {
      "type": "array",
      "items": {
        "type": "string",
        "format": "uri"
      },
      "title": "Individual Images"
    }
  }
}
        """
        return output

    def swap_face(self, image_face, image_target, output_folder="", nsfw=True):

        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

        input_params = {
            "swap_image": open(image_face, "rb"),
            "target_image": open(image_target, "rb"),
            'disable_safety_checker': nsfw
        }
        future = self.executor.submit(self.api.run, self.swap_model, input=input_params)
        output_url = future.result()
        output = []

        response = requests.get(output_url)
        file = os.path.join(output_folder, os.path.basename(image_target) + self.swap_path + ".png")
        with open(file, "wb") as f:
            f.write(response.content)
            output.append(file)

        return output

    def get_base64_image(self, image_path):
        if image_path is not None:
            # Ouvrir le fichier d'image
            with Image.open(image_path) as img:
                # Lire les octets de l'image
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                file_bytes = buffered.getvalue()

                # Encoder les octets en base64
                base64_image = base64.b64encode(file_bytes).decode('utf-8')
                img_value = f'data:image/jpeg;base64,{base64_image}'
                return img_value
        else:
            return None