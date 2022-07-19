import torch
import cv2
from config import NUM_CLASSES, IN_CHANNELS
from model import MRIModel
from argparse import ArgumentParser


def inference_model(model, image):
    with torch.no_grad():
        model.eval()
        btch = torch.transpose(torch.Tensor(image).unsqueeze(0), -1, 1)
        logits = model(btch)
        pr_mask = logits.sigmoid()
    return np.array(pr_mask.squeeze(0)[0, :, :])


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--backbone', type=str, required=True, help='Model backbone from segmentation-models-pytorch')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Input image')
    args = parser.parse_args()

    model = MRIModel(args.model, args.backbone, IN_CHANNELS, NUM_CLASSES)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    image = cv2.imread(args.image)

    pr_mask = inference_model(model, image)

    cv2.imshow('Image', image)
    cv2.imshow('Pred Mask', pr_mask)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
