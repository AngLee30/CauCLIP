import numpy

def generate_label(labels):
    num = len(labels)
    ground_truth = numpy.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                ground_truth[i, k] = 1
    return ground_truth

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def create_logits(x1, x2, logit_scale):
    """Cosine similarity similar to CLIP, use logit_scale from the pretrained CLIP model."""
    # normalized features
    x1 = x1 / x1.norm(dim=-1, keepdim=True)  # (bs, 512)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)  # (bs, 512)

    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logits_per_x1.t()

    return logits_per_x1, logits_per_x2