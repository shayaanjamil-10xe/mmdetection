#!/usr/bin/env python
# coding: utf-8

# # AutoQuant
# 
# This notebook shows a working code example of how to use AIMET AutoQuant feature.
# 
# AIMET offers a suite of neural network post-training quantization (PTQ) techniques that can be applied in succession. However, the process of finding the right combination and sequence of techniques to apply is time-consuming and requires careful analysis, which can be challenging especially for non-expert users. We instead recommend AutoQuant to save time and effort.
# 
# AutoQuant is an API that applies various PTQ techniques in AIMET automatically based on analyzing the model and best-known heuristics. In AutoQuant, users specify the amount of tolerable accuracy drop, and AutoQuant will apply PTQ techniques cumulatively until the target accuracy is satisfied.
# 
# 
# #### Overall flow
# This notebook covers the following
# 1. Define constants and helper functions
# 2. Load a pretrained FP32 model
# 3. Run AutoQuant
# 
# #### What this notebook is not
# This notebook is not designed to show state-of-the-art AutoQuant results. For example, it uses a relatively quantization-friendly model like Resnet18. Also, some optimization parameters are deliberately chosen to have the notebook execute more quickly.
# 

# ---
# ## Dataset
# 
# This notebook relies on the ImageNet dataset for the task of image classification. If you already have a version of the dataset readily available, please use that. Else, please download the dataset from appropriate location (e.g. https://image-net.org/challenges/LSVRC/2012/index.php#).
# 
# **Note1**: The ImageNet dataset typically has the following characteristics and the dataloader provided in this example notebook rely on these
# - Subfolders 'train' for the training samples and 'val' for the validation samples. Please see the [pytorch dataset description](https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html) for more details.
# - A subdirectory per class, and a file per each image sample
# 
# **Note2**: To speed up the execution of this notebook, you may use a reduced subset of the ImageNet dataset. E.g. the entire ILSVRC2012 dataset has 1000 classes, 1000 training samples per class and 50 validation samples per class. But for the purpose of running this notebook, you could perhaps reduce the dataset to say 2 samples per class. This exercise is left upto the reader and is not necessary.
# 
# Edit the cell below and specify the directory where the downloaded ImageNet dataset is saved.

# In[ ]:


import cv2
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import json
from mmcv.transforms import Compose
import numpy as np
from mmdet.utils import get_test_pipeline_cfg

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def preprocess(test_pipeline, image):
    if isinstance(image, np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)
    return test_pipeline(dict(img=image))

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_json_path, transform=None):
        self.transform = transform
        self.images_dir = images_dir
        self.annotations_json = read_json(annotations_json_path)


    def __len__(self):
        return len(self.annotations_json['images'])

    def __getitem__(self, idx):
        image_dict = self.annotations_json['images'][idx]
        image_path = os.path.join(self.images_dir, image_dict['file_name'])
        image_id = image_dict['id']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed_images = self.transform(image)
        else:
            transformed_images = image

        return image_id, image_path, transformed_images


# calibrationDataloader = DataLoader(calibrationDataset, batch_size=32, shuffle=True)


# ## 1. Define Constants and Helper functions
# 
# In this section the constants and helper functions needed to run this eaxmple are defined.
# 
# - **EVAL_DATASET_SIZE** A typical value is 5000. In this notebook, this value has been set to 500 for faster execution.
# - **CALIBRATION_DATASET_SIZE** A typical value is 2000. In this notebook, this value has been set to 200 for faster execution.
# 
# 
# The helper function **_create_sampled_data_loader()** returns a DataLoader based on the dataset and the number of samples provided.

# ## 2. Load a pretrained FP32 model
# For this example, we are going to load a pretrained resnet18 model from torchvision. Similarly, you can load any pretrained PyTorch model instead.

# In[ ]:


# %cd /content/drive/MyDrive/Aimet-torch/mmdetection-3.3.0/
import torch
from mmdet.apis import DetInferencer

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([640, 640]),  # Resize
])

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = 'rtmdet_tiny_8xb32-300e_coco.py'
WEIGHTS_PATH = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

ROOT_DATASET_DIR = '/teamspace/studios/this_studio/COCO'
IMAGES_DIR = os.path.join(ROOT_DATASET_DIR, 'images')
ANNOTATIONS_JSON_PATH = os.path.join(ROOT_DATASET_DIR, 'annotations/instances_val2017.json')
# ANNOTATIONS_JSON_PATH = "/home/shayaan/Desktop/aimet/my_mmdet/temp.json"

model = DetInferencer(model=CONFIG_PATH, weights=WEIGHTS_PATH, device=DEVICE)
evalDataset = CustomImageDataset(images_dir=IMAGES_DIR, annotations_json_path=ANNOTATIONS_JSON_PATH, transform=transform)


# In[ ]:


from mmcv.transforms import Compose
test_evaluator = model.cfg.test_evaluator
test_evaluator.type = 'mmdet.evaluation.CocoMetric' 
test_evaluator.dataset_meta = model.model.dataset_meta
test_evaluator.ann_file = ANNOTATIONS_JSON_PATH
test_evaluator = Compose(test_evaluator)


# In[ ]:


import random
from typing import Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from mmengine.structures import InstanceData

EVAL_DATASET_SIZE = 5000
CALIBRATION_DATASET_SIZE = 2000
BATCH_SIZE = 40
# _datasets = {}

# def _create_sampled_data_loader(dataset, num_samples):
#     if num_samples not in _datasets:
#         indices = random.sample(range(len(dataset)), num_samples)
#         _datasets[num_samples] = Subset(dataset, indices)
#     return DataLoader(_datasets[num_samples], batch_size=BATCH_SIZE)


# def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:    
#     data_loader = DataLoader(evalDataset, batch_size=BATCH_SIZE)

#     new_preds = []
#     for image_id, image_path, _ in tqdm(data_loader):
#         preds = model(image_path, out_dir='./output', batch_size=BATCH_SIZE)
        
#         for img_id, pred in zip(image_id, preds['predictions']):
#             new_pred = InstanceData(metainfo={"img_id": int(img_id)})
#             new_pred.bboxes = [np.array(p) for p in pred['bboxes']]
#             new_pred.labels = pred['labels']
#             new_pred.scores = pred['scores']
#             new_preds.append(new_pred)

#     eval_results = test_evaluator(new_preds)
#     with open("/home/shayaan/Desktop/aimet/my_mmdet/ptq_stats/eval_acc_fp32.json", "w") as f:
#         json.dump(eval_results, f, indent=4)
#     bbox_map = eval_results['bbox_mAP']
#     return bbox_map


# In[ ]:


from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.registry import MODELS

collate_preprocessor = model.preprocess
predict_by_feat = model.model.bbox_head.predict_by_feat
rescale = True

preprocessor = MODELS.build(model.cfg.model.data_preprocessor)
def add_pred_to_datasample(data_samples, results_list):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


# In[ ]:


def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:    
    data_loader = DataLoader(evalDataset, batch_size=BATCH_SIZE)

    new_preds = []
    for image_id, image_path, _ in tqdm(data_loader):
        pre_processed = collate_preprocessor(inputs=image_path, batch_size=BATCH_SIZE)
        _, data = list(pre_processed)[0]
        data = preprocessor(data, False)

        preds = model.model._forward(data['inputs'].cuda())
        batch_img_metas = [
        data_samples.metainfo for data_samples in data['data_samples']
        ]
        preds = predict_by_feat(*preds, batch_img_metas=batch_img_metas, rescale=True)
        preds = add_pred_to_datasample(data['data_samples'], preds)
        
        for img_id, pred in zip(image_id, preds):
            pred = pred.pred_instances
            new_pred = InstanceData(metainfo={"img_id": int(img_id)})
            new_pred.bboxes = [np.array(p) for p in pred['bboxes'].cpu()]
            new_pred.labels = pred['labels'].cpu()
            new_pred.scores = pred['scores'].cpu()
            new_preds.append(new_pred)
    
    eval_results = test_evaluator(new_preds)
    bbox_map = eval_results['bbox_mAP']
    with open("/teamspace/studios/this_studio/mmdetection/ptq_stats/eval_acc_fp32.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    return bbox_map


# In[ ]:


from aimet_torch.quantsim import QuantizationSimModel, QuantScheme
from aimet_torch.model_preparer import prepare_model

dummy_input = torch.rand(1, 3, 640, 640).cuda()


sim_model = prepare_model(model.model)

sim = QuantizationSimModel(model=sim_model,
                        quant_scheme=QuantScheme.post_training_tf_enhanced,
                        dummy_input=dummy_input,
                        default_output_bw=8,
                        default_param_bw=8,)


# In[ ]:


# print(str(sim))


# In[ ]:


print("CALCULATING FP32 ACCURACY NOW")
accuracy = eval_callback(model)


# In[ ]:


print(f'- FP32 accuracy: {accuracy}')


# ## 3. Run AutoQuant
# ### Create AutoQuant Object
# 
# The AutoQuant feature utilizes an unlabeled dataset to achieve quantization. The class **UnlabeledDatasetWrapper** creates an unlabeled Dataset object from a labeled Dataset.

# In[ ]:


class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset, num_samples):
        self._dataset = dataset
        self.num_samples = num_samples if num_samples < len(self._dataset) else len(self._dataset)
        self.transform = transform

    def __len__(self):
        return self.num_samples 

    def __getitem__(self, index):
        _, _, transformed_images = self._dataset[index]
        return transformed_images
    


# In[ ]:


from aimet_torch.auto_quant import AutoQuant
from glob import glob

def new_eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    data_loader = DataLoader(evalDataset, batch_size=BATCH_SIZE)
    new_preds = []
    for image_id, image_path, _ in tqdm(data_loader):
        pre_processed = collate_preprocessor(inputs=image_path, batch_size=BATCH_SIZE)
        _, data = list(pre_processed)[0]
        data = preprocessor(data, False)
        preds = model(data['inputs'].cuda())
        batch_img_metas = [
        data_samples.metainfo for data_samples in data['data_samples']
        ]
        preds = predict_by_feat(*preds, batch_img_metas=batch_img_metas, rescale=True)
        preds = add_pred_to_datasample(data['data_samples'], preds)
        
        for img_id, pred in zip(image_id, preds):
            pred = pred.pred_instances
            new_pred = InstanceData(metainfo={"img_id": int(img_id)})
            new_pred.bboxes = [np.array(p) for p in pred['bboxes'].cpu()]
            new_pred.labels = pred['labels'].cpu()
            new_pred.scores = pred['scores'].cpu()
            new_preds.append(new_pred)

    eval_results = test_evaluator(new_preds)
    num_file = len(glob("/teamspace/studios/this_studio/mmdetection/ptq_stats/eval_acc_quant_*"))
    with open(f"/teamspace/studios/this_studio/mmdetection/ptq_stats/eval_acc_quant_{num_file}.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    bbox_map = eval_results['bbox_mAP']
    return bbox_map


calibration_data_loader = DataLoader(UnlabeledDatasetWrapper(evalDataset, num_samples=CALIBRATION_DATASET_SIZE))
auto_quant = AutoQuant(sim_model,
                       dummy_input=dummy_input,
                       data_loader=calibration_data_loader,
                       eval_callback=new_eval_callback)


# ### Run AutoQuant Inference
# This step runs AutoQuant inference. AutoQuant inference will run evaluation using the **eval_callback** with the vanilla quantized model without applying PTQ techniques. This will be useful for measuring the baseline evaluation score before running AutoQuant optimization.

# In[ ]:


sim, initial_accuracy = auto_quant.run_inference()


# In[ ]:


print(f"- Quantized A ccuracy (before optimization): {initial_accuracy}")


# ### Set AdaRound Parameters (optional)
# AutoQuant uses a set of predefined default parameters for AdaRound.
# These values were determined empirically and work well with the common models.
# However, if necessary, you can also use your custom parameters for Adaround.
# In this notebook, we will use very small AdaRound parameters for faster execution.

# In[ ]:


from aimet_torch.adaround.adaround_weight import AdaroundParameters

adaround_params = AdaroundParameters(calibration_data_loader, num_batches=len(calibration_data_loader), default_num_iterations=2000)
auto_quant.set_adaround_params(adaround_params)


# ### Run AutoQuant Optimization
# This step runs AutoQuant optimization, which returns the best possible quantized model, corresponding evaluation score and the path to the encoding file.
# The **allowed_accuracy_drop** parameter indicates the tolerable amount of accuracy drop. AutoQuant applies a series of quantization features until the target accuracy (FP32 accuracy - allowed accuracy drop) is satisfied. When the target accuracy is reached, AutoQuant will return immediately without applying furhter PTQ techniques. Please refer AutoQuant User Guide and API documentation for complete details.

# In[ ]:


model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy}")


# In[ ]:


# model, optimized_accuracy, encoding_path


# ---
# ## Summary
# 
# Hope this notebook was useful for you to understand how to use AIMET AutoQuant feature.
# 
# Few additional resources
# - Refer to the AIMET API docs to know more details of the APIs and parameters
# - Refer to the other example notebooks to understand how to use AIMET CLE and AdaRound features in a standalone fashion.
