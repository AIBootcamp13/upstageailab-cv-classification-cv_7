import io
import base64
import io
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Input:
      tensor: shape [C, H, W], normalized
    Output:
      denormalized tensor (still float, 0~1)
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def tensor_to_thumbnail_base64(tensor, size=(128, 128), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    tensor = denormalize(tensor, mean, std)
    pil_img = to_pil_image(tensor)
    pil_img = pil_img.resize(size, Image.BILINEAR)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(base64_str):
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

