import os
from time import sleep
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import threading
from MobileSamModel.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./MobileSamModel/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
current_frame = frame
masks = []



stop_threads = True
sleep(1)
stop_threads = False

mask_ind = 0

def updateMasks():
    global masks
    global mask_ind
    global current_frame
    while True:
        global stop_threads
        if stop_threads:
            break
        predictor.set_image(current_frame)
        # predictor.set_image(image)
        # tmasks, _, _ = predictor.predict()  
        input_point = np.array([[250, 375]])
        input_label = np.array([1])

        tmasks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )          
        if len(tmasks) == 0:
            # print("No masks found")
            continue            
        if mask_ind >= len(tmasks):
            mask_ind = len(tmasks) - 1
        
        tmasks = tmasks[mask_ind].astype(np.uint8)
        # turn 0 and 1 to 0 and 255
        tmasks = tmasks * 255
        # turn into 3 channels
        masks = np.stack([tmasks] * 3, axis=-1)
        
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def updateFrame():
    global current_frame
    global cap
    while True:
        global stop_threads
        if stop_threads:
            break
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame = frame        
    
        if not ret:
            print("Error: failed to capture image")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def showMask():
    global masks
    while True:
        global stop_threads
        if stop_threads:
            break
        if len(masks) == 0:
            # print("No masks found")
            continue
        cv2.imshow("masks", masks)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def runCam():

    mask_ind = 0
    t1 = threading.Thread(target=updateFrame)
    t1.start()
    sleep(1)
    t2 = threading.Thread(target=updateMasks)
    t2.start()
    sleep(1)
    t3 = threading.Thread(target=showMask)
    t3.start()

    while True:
        # get mask index from user
        inp = input("Enter mask index: ")
        mask_ind = int(inp)
       

# print(os.getcwd())
def basicSamRun():
    
    image = cv2.imread('./MobileSamModel/notebooks/images/picture2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

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

    # masks, _, low_res_logits = predictor.predict(None, ort_inputs)
    # # masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    # masks = masks > predictor.model.mask_threshold

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print(masks.shape)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  

    # plt.figure(figsize=(10,10))
    # plt.imshow(masks[1])
    # # show_mask(masks, plt.gca())
    # # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show() 
        

runCam()