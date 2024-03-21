import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from MobileSamModel.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./MobileSamModel/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

image = cv2.imread('./MobileSamModel/notebooks/picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = SamPredictor(mobile_sam)
predictor.set_image(image)

input_point = np.array([[250, 375]])
input_label = np.array([1])

ort_inputs = {
    "image_embeddings": predictor.get_image_embedding().cpu().numpy(),
    "point_coords": np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :],
    "point_labels": np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32),
    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
    "has_mask_input": np.zeros(1, dtype=np.float32),
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}
masks, _, _ = predictor.predict(ort_inputs)

print(masks.shape)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# # show_mask(masks, plt.gca())
# # show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.show() 