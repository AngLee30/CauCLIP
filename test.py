import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from modules.fusion_model import FusionModel

def validate(
        val_loader: DataLoader, 
        classes: torch.Tensor, 
        device: str, 
        model: clip.CLIP, 
        fusion_model: FusionModel, 
        config, 
        num_text_aug: int
    ):
    model.eval()
    fusion_model.eval()

    sample_num = 0
    corr_cnt = [0, 0, 0]
    total_cnt = [0, 0, 0]

    with torch.no_grad():
        text_inputs = classes.to(device) # (num_text_aug*3, 77)              
        text_features = model.encode_text(text_inputs) # (num_text_aug*3, 512)

        for iteration, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.view((-1, config.data.seg_num, 3) + images.size()[-2:]) # (bs, T, 3, 224, 224)
            b, t, c, h, w = images.size()
            labels = labels.to(device) # labels: [label_1, label_2, ... , label_bs] (list)
            image_input = images.to(device).view(-1, c, h, w) # (bs*T, 3, 224, 224)
            image_features = model.encode_image(image_input) # (bs*T, 512)
            image_features = image_features.view(b, t, -1) # (bs, T, 512)
            image_features = fusion_model(image_features) # (bs, 512)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (bs, 512)                      
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (num_text_aug*3, 512)                       
            similarity = (100.0 * image_features @ text_features.t()) # (bs, num_text_aug*3)                          
            similarity = similarity.view(b, num_text_aug, -1) # (bs, num_text_aug, 3)                                 

            similarity = similarity.softmax(dim=-1) # (bs, num_text_aug, 3)
            similarity = similarity.mean(dim=1, keepdim=False) # (bs, 3)                               
            values_1, indices_1 = similarity.topk(1, dim=-1) # indices_1: (bs, 1)                       
            sample_num += b
            
            for i in range(b): # process a batch
                label = int(labels[i])
                predict = int(indices_1[i])
                total_cnt[label] += 1
                if predict == label:
                    corr_cnt[label] += 1
            
        acc_0 = corr_cnt[0] / total_cnt[0] if total_cnt[0] != 0 else 0
        acc_1 = corr_cnt[1] / total_cnt[1] if total_cnt[1] != 0 else 0
        acc_2 = corr_cnt[2] / total_cnt[2] if total_cnt[2] != 0 else 0
        print(f"acc_0: {acc_0}, acc_1: {acc_1}, acc_2: {acc_2}")
        bacc = (acc_0 + acc_1 + acc_2) / 3.0 # balanced accuracy
        return bacc