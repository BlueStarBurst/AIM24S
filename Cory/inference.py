from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import os
from time import sleep

model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

abspath = os.path.abspath(__file__)

image = cv2.imread(os.path.join(os.path.dirname(abspath), "test_imgs/labrador.jpg"))
# turn into ndarray
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = SamPredictor(mobile_sam)

while True:
    print("predicting...")
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    # save mask
    # print(masks)
    # turn bool to int
    masks = masks.astype(int)
    # turn 0 and 1 to 0 and 255
    masks = masks * 255
    cv2.imwrite(os.path.join(os.path.dirname(abspath), "test_imgs/mask.jpg"), masks[0])
    print("done")