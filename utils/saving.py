import torch

def epoch_saving(file_name, epoch, model, fusion_model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'fusion_model_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 
        file_name
    )

def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'fusion_model_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 
        best_name
    )