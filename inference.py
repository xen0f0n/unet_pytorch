import cv2
import torch
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from models.unet_orig import UNetOrig as net
from utils.json_parser import parse_json


def infer(config_params, ckpt_path, test_file_paths_npy, save_folder):
    device = config_params["trainer"]["device"]
    input_channels = config_params["trainer"]["in_channels"]
    output_classes = config_params["trainer"]["out_channels"]

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)

    model = net(n_channels=input_channels, n_classes=output_classes).to(device)
    model.to(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model.eval()

    files = np.load(test_file_paths_npy)
    for file in files:
        filename, _ = os.path.splitext(file.split('/')[-1])

        img = cv2.imread(file, 0)
        img = (img - img.mean()) / img.std()
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        X = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(X)

        probs = torch.sigmoid(output)
        probs = torch.clamp(probs, 0.0, 1.0)
        probs_as_np = probs.squeeze().cpu().numpy()
        torch.cuda.empty_cache()

        probs_as_np = (probs_as_np * 255).astype('uint8')
        cv2.imwrite(os.path.join(save_folder, f'{filename}_pred.tif'), probs_as_np)
        # plt.imshow(probs_as_np)
        # plt.show()

        pred_thresh, pred_bin = cv2.threshold(probs_as_np, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(save_folder, f'{filename}_pred_bin.tif'), pred_bin)
        # plt.imshow(pred_bl_bin)
        # plt.show()


if __name__ == '__main__':

    config_params = parse_json('./experiments/demo/config.json')
    ckpt_path = './experiments/demo/checkpoints/demo.pth'
    test_file_paths_npy = './data/membrane/test_files.npy'
    save_folder = './experiments/demo/predictions'

    infer(config_params, ckpt_path, test_file_paths_npy, save_folder)
