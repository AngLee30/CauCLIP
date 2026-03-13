import os
import time
import argparse
import shutil
from pathlib import Path

import yaml
import numpy
import torch
from torch.nn.parallel import DataParallel
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotmap import DotMap

import clip
from modules.encoder import ImageEncoder, TextEncoder
from modules.fusion_model import FusionModel
from datasets.datasets import SurgVisDom
from datasets.augmentation import get_augmentation, rand_augment
from utils.loss import KLLoss, suppressionLoss
from utils.tools import generate_label, convert_models_to_fp32, create_logits
from utils.text_prompt import text_prompt
from utils.saving import best_saving, epoch_saving
from test import validate

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['arch'], time.strftime("%Y%m%d-%H%M", time.localtime()))

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    transform_train = get_augmentation(training=True, config=config)
    transform_val = get_augmentation(training=False, config=config)
    if config.data.randaug.enabled:
        print("Use rand_augment in transform_train.")
        transform_train = rand_augment(transform_train, config)
    
    device = "cuda" # training only on GPUs
    model, state_dict = clip.load(name=config.network.arch, device=device, jit=False) # load pretrained CLIP model

    fusion_model = FusionModel(state_dict, config) # type: ignore
    text_encoder = TextEncoder(model)
    image_encoder = ImageEncoder(model)

    clip.model.convert_weights(text_encoder)
    clip.model.convert_weights(image_encoder)

    fusion_model = DataParallel(fusion_model).cuda()
    text_encoder = DataParallel(text_encoder).cuda()
    image_encoder = DataParallel(image_encoder).cuda()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            fusion_model.module.load_state_dict(checkpoint['fusion_model_state_dict'], strict=True)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            fusion_model.module.load_state_dict(checkpoint['fusion_model_state_dict'], strict=True)
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    train_data = SurgVisDom(
        list_file=config.data.train_list, 
        labels_file=config.data.label_list,
        root_dir=config.data.train_root_dir,
        seg_num=config.data.seg_num,
        seg_length=config.data.seg_length,
        image_tmpl=config.data.image_tmpl,
        transform=transform_train,
        random_shift=config.data.random_shift,
        index_bias=config.data.index_bias,
        training_mode=True,
        alpha=config.data.alpha
    )

    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=config.data.batch_size,
        num_workers=1, 
        shuffle=True, 
        pin_memory=False, 
        drop_last=True
    )

    val_data = SurgVisDom(
        list_file=config.data.val_list, 
        labels_file=config.data.label_list,
        root_dir=config.data.val_root_dir,
        seg_num=config.data.seg_num,
        seg_length=config.data.seg_length,
        image_tmpl=config.data.image_tmpl, 
        transform=transform_val,
        random_shift=False,
        index_bias=config.data.index_bias,
        training_mode=False
    )

    val_loader = DataLoader(
        dataset=val_data, 
        batch_size=config.data.batch_size,
        num_workers=1, 
        shuffle=False, 
        pin_memory=False, 
        drop_last=False
    )

    classes, num_text_aug, text_dict = text_prompt(data=train_data, prompt_type=config.data.prompt_type)
    # classes: (num_text_aug*3, 77) (torch.Tensor)
    # text_dict: num_text_aug keys, each corresponds to a (3, 77) tensor

    # initialize optimizer
    image_encoder_params = list(map(id, model.visual.parameters()))
    text_encoder_params = filter(lambda p: id(p) not in image_encoder_params, model.parameters())
    optimizer = AdamW(
        [
            {'params': text_encoder_params},
            {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
            {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}
        ],
        betas=(0.9, 0.98), 
        lr=config.solver.lr, 
        eps=1e-8,
        weight_decay=config.solver.weight_decay
    )
    
    # initialize lr_scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.solver.epochs * len(train_loader), # update learning rate per iteration
        eta_min=1e-6,
        last_epoch=start_epoch * len(train_loader) - 1
    )
    
    kl_loss = KLLoss()
    suppression_loss = suppressionLoss()

    lambda_aug = config.solver.lambda_aug
    lambda_sup = config.solver.lambda_sup

    best_bacc = 0.0
    best_epoch = -1
    best_list = []

    for epoch in range(start_epoch, config.solver.epochs): # per epoch
        epoch_loss = 0.0
        image_encoder.train()
        text_encoder.train()
        fusion_model.train()
        for iteration, (images, images_aug, labels) in enumerate(tqdm(train_loader)): # per iteration
            # images: (bs, T*3, 224, 224) (torch.Tensor)
            # images_aug: (bs, T*3, 224, 224) (torch.Tensor)
            # labels: [label_1, label_2, ... , label_bs] (list)
            optimizer.zero_grad()

            # ================ generate embedding for original and augmented video ================
            images = images.view((-1, config.data.seg_num, 3) + images.size()[-2:]) # (bs, T, 3, 224, 224)
            images_aug = images_aug.view((-1, config.data.seg_num, 3) + images.size()[-2:]) # (bs, T, 3, 224, 224)

            b, t, c, h, w = images.size()
            images = images.to(device).view(-1, c, h, w) # (bs*T, 3, 224, 224)
            images_aug = images_aug.to(device).view(-1, c, h, w) # (bs*T, 3, 224, 224)

            image_embedding = image_encoder(images) # (bs*T, 512)
            image_embedding = image_embedding.view(b, t, -1) # (bs, T, 512)
            orig_video_embedding = fusion_model(image_embedding) # (bs, 512)
            assert torch.isfinite(orig_video_embedding).all()

            image_aug_embedding = image_encoder(images_aug) # (bs*T, 512)
            image_aug_embedding = image_aug_embedding.view(b, t, -1) # (bs, T, 512)
            aug_video_embedding = fusion_model(image_aug_embedding) # (bs, 512)
            assert torch.isfinite(aug_video_embedding).all()

            #x = orig_video_embedding
            #x = x / x.norm(dim=-1, keepdim=True)
            #y = aug_video_embedding
            #y = y / y.norm(dim=-1, keepdim=True)
            #print(x @ y.t()) # cosine similarity matrix
            #exit()
            # Above is a simple unit test, if no augmentation is implemented (alpha = 0.0),
            # the values on the main diagonal of the cosine similarity matrix should be all close to 1.

            # ================ generate embedding for text ================
            text_id = numpy.random.randint(num_text_aug, size=len(labels)) # numpy.ndarray
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(labels, text_id)]) # (bs, 77)

            texts = texts.to(device) # (bs, 77)
            text_embedding = text_encoder(texts) # (bs, 512)

            # ================ calculate cosine similarity as logits ================
            logit_scale = model.logit_scale
            logits_orig_video, logits_orig_txt = create_logits(orig_video_embedding, text_embedding, logit_scale)
            logits_aug_video, logits_aug_txt = create_logits(aug_video_embedding, text_embedding, logit_scale)

            ground_truth = torch.tensor(generate_label(labels), dtype=image_embedding.dtype, device=device)

            loss_orig_video = kl_loss(logits_orig_video, ground_truth)
            loss_orig_txt = kl_loss(logits_orig_txt, ground_truth)
            L_orig = (loss_orig_video + loss_orig_txt) / 2.0

            loss_aug_video = kl_loss(logits_aug_video, ground_truth)
            loss_aug_txt = kl_loss(logits_aug_txt, ground_truth)
            L_aug = (loss_aug_video + loss_aug_txt) / 2.0

            L_sup = suppression_loss(orig_video_embedding, aug_video_embedding)
            total_loss = L_orig +  lambda_aug * L_aug + lambda_sup * L_sup
            total_loss.backward()
            epoch_loss += total_loss.item()
        
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            lr_scheduler.step()

        print(f'Current epoch: {epoch}, epoch loss: {epoch_loss:.4f}.')

        if (epoch + 1) % config.logging.eval_freq == 0 or epoch == 0:
            bacc = validate(
                val_loader=val_loader, 
                classes=classes, 
                device=device, 
                model=model, 
                fusion_model=fusion_model.module,  # type: ignore
                config=config, 
                num_text_aug=num_text_aug
            )
            if bacc > best_bacc:
                best_bacc = bacc
                best_epoch = epoch
                best_list.append(best_epoch)
                best_saving(working_dir, epoch, model, fusion_model.module, optimizer)
                print(f'best balanced accuracy: {bacc:.4f}')
                print(f'best epoch: {best_epoch}')
                print(f'best list: {best_list}')

        file_name = "{}/last_model.pt".format(working_dir)
        epoch_saving(file_name, epoch, model, fusion_model.module, optimizer)

if __name__ == '__main__':
    main()