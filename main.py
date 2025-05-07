from train import train
from test import test_model
import torch
from model import swin_model_fpn
num_classes = 5
model_save_path = "swin_model_fpn+head.pth"
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        model = swin_model_fpn(num_classes, device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        test_model(device, model)
