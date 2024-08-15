import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Extract features from image1
# image1 = Image.open('img1.jpg')
image1 = Image.open('/root/nips/diffusers/examples/datasets/yann/yan5.jpg')
with torch.no_grad():
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    image_features1 = model.get_image_features(**inputs1)

#Extract features from image2
image2 = Image.open('/root/nips/diffusers/examples/loldu/Picture14.png')
with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    image_features2 = model.get_image_features(**inputs2)

# Extract features from text
text = "A description of the image"  # Replace with your text description
with torch.no_grad():
    text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)


#Compute their cosine similarity and convert it into a score between 0 and 1
cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
sim = (sim+1)/2
print('Image Similarity:', sim)


# Compute their cosine similarity and convert it into a score between 0 and 1
cos = nn.CosineSimilarity(dim=1)
sim = cos(image_features2, text_features).item()
sim = (sim + 1) / 2
print('Image-Text Similarity:', sim)




processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)


with torch.no_grad():
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    outputs1 = model(**inputs1)
    image_features1 = outputs1.last_hidden_state
    image_features1 = image_features1.mean(dim=1)

with torch.no_grad():
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    outputs2 = model(**inputs2)
    image_features2 = outputs2.last_hidden_state
    image_features2 = image_features2.mean(dim=1)

cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features1[0],image_features2[0]).item()
sim = (sim+1)/2
print('Dino Similarity:', sim)


