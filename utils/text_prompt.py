import torch
import clip

def text_prompt(data, prompt_type):
    cap_prompt = [f"{{}}"] # 1

    gen_prompt = [f"a photo of {{}}",
                  f"a picture of action {{}}",
                  f"Playing action of {{}}",
                  f"Playing a kind of action, {{}}",
                  f"Doing a kind of action, {{}}",
                  f"Can you recognize the action of {{}}?",
                  f"Video classification of {{}}",
                  f"A video of {{}}"] # 8

    cus_prompt = [f"a photo of {{}}, a type of surgical action",
                  f"Surgical action of {{}}",
                  f"{{}}, a surgical action",
                  f"{{}}, this is a surgical action",
                  f"{{}}, a video of surgical action",
                  f"Look, the surgeon is {{}}",
                  f"The doctor is performing {{}}",
                  f"The surgeon is performing {{}}"] # 8

    aug_prompt = [f"a photo of {{}}, a type of surgical action",
                f"a picture of action {{}}",
                f"Surgical action of {{}}",
                f"{{}}, an action",
                f"{{}} this is an action",
                f"{{}}, a video of surgical action",
                f"Playing action of {{}}",
                f"{{}}",
                f"Playing a kind of action, {{}}",
                f"Doing a kind of action, {{}}",
                f"Look, the surgeon is {{}}",
                f"Can you recognize the action of {{}}?",
                f"Video classification of {{}}",
                f"A video of {{}}",
                f"The doctor is performing {{}}",
                f"The surgeon is performing {{}}"] # 16

    if prompt_type == 0:
        text_aug = list(cap_prompt)
    elif prompt_type == 1:
        text_aug = list(gen_prompt)
    elif prompt_type == 2:
        text_aug = list(cus_prompt)
    else:
        text_aug = list(aug_prompt)

    text_dict: dict = {}
    num_text_aug: int = len(text_aug)

    for i, txt in enumerate(text_aug):
        # CLIP has a fixed token length for text, which is 77
        # data.classes: list: [[0, '...'], [1, '...'], [2, '...']]
        text_dict[i] = torch.cat([clip.tokenize(txt.format(content)) for idx, content in data.classes], dim=0) # (3, 77)
        assert text_dict[i].shape[0] == len(data.classes) and text_dict[i].shape[1] == 77

    classes: torch.Tensor = torch.cat([value for key, value in text_dict.items()]) # (num_text_aug*3, 77)

    return classes, num_text_aug, text_dict