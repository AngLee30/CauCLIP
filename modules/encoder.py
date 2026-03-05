import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model) :
        super(TextEncoder, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageEncoder(nn.Module):
    def __init__(self, model) :
        super(ImageEncoder, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)