REPLICATE PORTRAIT

import { writeFile } from "fs/promises";
import Replicate from "replicate";
const replicate = new Replicate();

const input = {
    background: "black",
    num_images: 13,
    input_image: "https://replicate.delivery/pbxt/N5DXcBZiATNE0n0Wu7ghgVh5i7VoNzzfYtyGoNdbKYnZic7L/replicate-prediction-f2d25rg6gnrma0cq257vdw2n4c.png",
    randomize_images: true
};

const output = await replicate.run("flux-kontext-apps/portrait-series", { input });

// To access the file URLs:
console.log(output[0].url());
//=> "https://replicate.delivery/.../output_0.png"

// To write the files to disk:
for (const [index, item] of Object.entries(output)) {
  await writeFile(`output_${index}.png`, item);
}
//=> output_0.png, output_1.png, output_2.png, output_3.png, o...


FLUX KONTEXT ON REPLICATE:
import replicate

input = {
    "prompt": "Make this a 90s cartoon",
    "input_image": "https://replicate.delivery/pbxt/N55l5TWGh8mSlNzW8usReoaNhGbFwvLeZR3TX1NL4pd2Wtfv/replicate-prediction-f2d25rg6gnrma0cq257vdw2n4c.png",
    "output_format": "jpg"
}

output = replicate.run(
    "black-forest-labs/flux-kontext-pro",
    input=input
)

# To access the file URL:
print(output.url())
#=> "https://replicate.delivery/.../output.jpg"

# To write the file to disk:
with open("output.jpg", "wb") as file:
    file.write(output.read())
#=> output.jpg written to disk



REPLICATE FLUX KONTEXT COMPOSE
import replicate

input = {
    "prompt": "Combine these photos into one fluid scene. Make the woman in the first image wear the futuristic headgear in the second image. Then put the same woman the third image's scene.",
    "aspect_ratio": "1:1",
    "input_images": ["https://replicate.delivery/pbxt/N83LmkC1NAWFfeIkF6HBSkQGN2W3tr6Q7XIxhRcA1Eoh0uAC/tmp04o2q0cj.png","https://replicate.delivery/pbxt/N83LmgcB5EguqxLt5gQnmI2LXRa8H0eayIrtVw931df2xvtt/test_4.jpg","https://replicate.delivery/pbxt/N83Ln0m0xhzTsR5rY4RAREEpPLwzfXSB90Fmp24gQFHKoQ2D/test.jpg"]
}

output = replicate.run(
    "flux-kontext-apps/multi-image-list",
    input=input
)

# To access the file URL:
print(output.url())
#=> "https://replicate.delivery/.../output.png"

# To write the file to disk:
with open("output.png", "wb") as file:
    file.write(output.read())
#=> output.png written to disk



REPLICATE LLAVA
import replicate

input = {
    "image": "https://replicate.delivery/pbxt/KRULC43USWlEx4ZNkXltJqvYaHpEx2uJ4IyUQPRPwYb8SzPf/view.jpg",
    "prompt": "Are you allowed to swim here?"
}

for event in replicate.stream(
    "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    input=input
):
    print(event, end="")
    #=> "Yes, "

