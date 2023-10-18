"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import skimage.draw
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################


class AbnormalityConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "abnormality"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################


#  Dataset
############################################################

class AbnormalityDataset(utils.Dataset):
    def load_abnormalities(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                           class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        self.add_class("abnormality", 1, "benign_cyst_neoplasia")
        self.add_class("abnormality", 2, "malignant_neoplasia")
        self.add_class("abnormality", 3, "inflammation")
        self.add_class("abnormality", 4, "dysplasia")
        self.add_class("abnormality", 5, "metabolic/systemic")
        self.add_class("abnormality", 6, "trauma")
        self.add_class("abnormality", 7, "developmental")

        if subset == 'train':
            annot_dir = os.path.join(dataset_dir, 'train.json')
        else:
            annot_dir = os.path.join(dataset_dir, 'val.json')

        annotations = json.load(open(annot_dir))

        for annot in annotations:
            # get external id, polygons, class
            self.add_image('abnormality',
                           image_id=annot['image_id'],
                           path=annot['path'],
                           width=annot['width'],
                           height=annot['height'],
                           num_ids=annot['num_ids'],
                           polygons=annot['polygons'],
                           # chars=annot['chars'],
                           # description=annot['description']
                           )

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "abnormality":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, list_p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1

            for p in list_p:
                p_mask = skimage.draw.polygon2mask(image_shape=(info["height"], info["width"]),
                                                   polygon=[[y, x] for x, y in p]).astype(np.uint8)
                mask[:, :, i] = np.logical_or(mask[:, :, i], p_mask)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "abnormality":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    dataset_dir = os.path.join(ROOT_DIR, 'dataset')
    # Training dataset.
    dataset_train = AbnormalityDataset()
    dataset_train.load_abnormalities(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AbnormalityDataset()
    dataset_val.load_abnormalities(dataset_dir, "val")
    dataset_val.prepare()

    # Define your augmentation sequence
    augmentation = imgaug.augmenters.Sequential([
        imgaug.augmenters.Fliplr(0.5), # Horizontal flips with a 50% probability
        imgaug.augmenters.Sometimes(.5, imgaug.augmenters.contrast.LinearContrast((0.5, 2.0))), # Adjust contrast
        imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Crop(percent=(0.0, 0.1))), # Randomly crop the image
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=augmentation)

#     def auto_download(self, dataDir, dataType, dataYear):
#         """Download the COCO dataset/annotations if requested.
#         dataDir: The root directory of the COCO dataset.
#         dataType: What to load (train, val, minival, valminusminival)
#         dataYear: What dataset year to load (2014, 2017) as a string, not an integer
#         Note:
#             For 2014, use "train", "val", "minival", or "valminusminival"
#             For 2017, only "train" and "val" annotations are available
#         """
#
#         # Setup paths and file names
#         if dataType == "minival" or dataType == "valminusminival":
#             imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
#         else:
#             imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
#         # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)
#
#         # Create main folder if it doesn't exist yet
#         if not os.path.exists(dataDir):
#             os.makedirs(dataDir)
#
#         # Download images if not available locally
#         if not os.path.exists(imgDir):
#             os.makedirs(imgDir)
#             print("Downloading images to " + imgZipFile + " ...")
#             with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
#                 shutil.copyfileobj(resp, out)
#             print("... done downloading.")
#             print("Unzipping " + imgZipFile)
#             with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
#                 zip_ref.extractall(dataDir)
#             print("... done unzipping")
#         print("Will use images in " + imgDir)
#
#         # Setup annotations data paths
#         annDir = "{}/annotations".format(dataDir)
#         if dataType == "minival":
#             annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_minival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
#             unZipDir = annDir
#         elif dataType == "valminusminival":
#             annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_valminusminival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
#             unZipDir = annDir
#         else:
#             annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
#             annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
#             annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
#             unZipDir = dataDir
#         # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)
#
#         # Download annotations if not available locally
#         if not os.path.exists(annDir):
#             os.makedirs(annDir)
#         if not os.path.exists(annFile):
#             if not os.path.exists(annZipFile):
#                 print("Downloading zipped annotations to " + annZipFile + " ...")
#                 with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
#                     shutil.copyfileobj(resp, out)
#                 print("... done downloading.")
#             print("Unzipping " + annZipFile)
#             with zipfile.ZipFile(annZipFile, "r") as zip_ref:
#                 zip_ref.extractall(unZipDir)
#             print("... done unzipping")
#         print("Will use annotations in " + annFile)
#         # The following two functions are from pycocotools with a few changes.
#
#         def annToRLE(self, ann, height, width):
#             """
#             Convert annotation which can be polygons, uncompressed RLE to RLE.
#             :return: binary mask (numpy 2D array)
#             """
#             segm = ann['segmentation']
#             if isinstance(segm, list):
#                 # polygon -- a single object might consist of multiple parts
#                 # we merge all parts into one mask rle code
#                 rles = maskUtils.frPyObjects(segm, height, width)
#                 rle = maskUtils.merge(rles)
#             elif isinstance(segm['counts'], list):
#                 # uncompressed RLE
#                 rle = maskUtils.frPyObjects(segm, height, width)
#             else:
#                 # rle
#                 rle = ann['segmentation']
#             return rle
#
#         def annToMask(self, ann, height, width):
#             """
#             Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#             :return: binary mask (numpy 2D array)
#             """
#             rle = self.annToRLE(ann, height, width)
#             m = maskUtils.decode(rle)
#             return m
#
#
# ############################################################
# #  COCO Evaluation
# ############################################################
#
# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#     """Arrange resutls to match COCO specs in http://cocodataset.org/#format
#     """
#     # If no results, return an empty list
#     if rois is None:
#         return []
#
#     results = []
#     for image_id in image_ids:
#         # Loop through detections
#         for i in range(rois.shape[0]):
#             class_id = class_ids[i]
#             score = scores[i]
#             bbox = np.around(rois[i], 1)
#             mask = masks[:, :, i]
#
#             result = {
#                 "image_id": image_id,
#                 "category_id": dataset.get_source_class_id(class_id, "coco"),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                 "score": score,
#                 "segmentation": maskUtils.encode(np.asfortranarray(mask))
#             }
#             results.append(result)
#     return results
#
#
# def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
#     """Runs official COCO evaluation.
#     dataset: A Dataset object with valiadtion data
#     eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
#     limit: if not 0, it's the number of images to use for evaluation
#     """
#     # Pick COCO images from the dataset
#     image_ids = image_ids or dataset.image_ids
#
#     # Limit to a subset
#     if limit:
#         image_ids = image_ids[:limit]
#
#     # Get corresponding COCO image IDs.
#     coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
#
#     t_prediction = 0
#     t_start = time.time()
#
#     results = []
#     for i, image_id in enumerate(image_ids):
#         # Load image
#         image = dataset.load_image(image_id)
#
#         # Run detection
#         t = time.time()
#         r = model.detect([image], verbose=0)[0]
#         t_prediction += (time.time() - t)
#
#         # Convert results to COCO format
#         # Cast masks to uint8 because COCO tools errors out on bool
#         image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
#                                            r["rois"], r["class_ids"],
#                                            r["scores"],
#                                            r["masks"].astype(np.uint8))
#         results.extend(image_results)
#
#     # Load results. This modifies results with additional attributes.
#     coco_results = coco.loadRes(results)
#
#     # Evaluate
#     cocoEval = COCOeval(coco, coco_results, eval_type)
#     cocoEval.params.imgIds = coco_image_ids
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#     print("Prediction time: {}. Average {}/image".format(
#         t_prediction, t_prediction / len(image_ids)))
#     print("Total time: ", time.time() - t_start)
#
#
# ############################################################
# #  Training
# ############################################################


if __name__ == '__main__':
    # load configuration
    config = AbnormalityConfig()
    # load model
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=DEFAULT_LOGS_DIR)
    # load weights
    weights_path = COCO_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    # train model
    train(model)



# import argparse
    #
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN on MS COCO.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'evaluate' on MS COCO")
    # parser.add_argument('--dataset', required=True,
    #                     metavar="/path/to/coco/",
    #                     help='Directory of the MS-COCO dataset')
    # parser.add_argument('--year', required=False,
    #                     default=DEFAULT_DATASET_YEAR,
    #                     metavar="<year>",
    #                     help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    # parser.add_argument('--model', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--limit', required=False,
    #                     default=500,
    #                     metavar="<image count>",
    #                     help='Images to use for evaluation (default=500)')
    # parser.add_argument('--download', required=False,
    #                     default=False,
    #                     metavar="<True|False>",
    #                     help='Automatically download and unzip MS-COCO files (default=False)',
    #                     type=bool)
    # args = parser.parse_args()
    # print("Command: ", args.command)
    # print("Model: ", args.model)
    # print("Dataset: ", args.dataset)
    # print("Year: ", args.year)
    # print("Logs: ", args.logs)
    # print("Auto Download: ", args.download)
    #
    # # Configurations
    # if args.command == "train":
    #     config = AbnormalityConfig()
    # else:
    #     class InferenceConfig(AbnormalityConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #         DETECTION_MIN_CONFIDENCE = 0
    #     config = InferenceConfig()
    # config.display()
    #
    # # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    #
    # # Select weights file to load
    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    # if args.model.lower() == "coco":
    #     model_path = COCO_MODEL_PATH
    # elif args.model.lower() == "last":
    #     # Find last trained weights
    #     model_path = model.find_last()
    # elif args.model.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     model_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights
    #
    #
    # # Load weights
    # print("Loading weights ", weights_path)
    # model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    # if args.command == 'train':
    #     train(model)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'evaluate'".format(args.command))
    # if args.command == "train":
    #     # Training dataset. Use the training set and 35K from the
    #     # validation set, as as in the Mask RCNN paper.
    #     dataset_train = AbnormalityDataset()
    #     dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
    #     dataset_train.prepare()
    #
    #     # Validation dataset
    #     dataset_val = AbnormalityDataset()
    #     val_type = "val" if args.year in '2017' else "minival"
    #     dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
    #     dataset_val.prepare()
    #
    #     # Image Augmentation
    #     # Right/Left flip 50% of the time
    #     augmentation = imgaug.augmenters.Fliplr(0.5)
    #
    #     # *** This training schedule is an example. Update to your needs ***
    #
    #     # Training - Stage 1
    #     print("Training network heads")
    #     model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE,
    #                 epochs=40,
    #                 layers='heads',
    #                 augmentation=augmentation)
    #
    #     # Training - Stage 2
    #     # Finetune layers from ResNet stage 4 and up
    #     print("Fine tune Resnet stage 4 and up")
    #     model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE,
    #                 epochs=120,
    #                 layers='4+',
    #                 augmentation=augmentation)
    #
    #     # Training - Stage 3
    #     # Fine tune all layers
    #     print("Fine tune all layers")
    #     model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE / 10,
    #                 epochs=160,
    #                 layers='all',
    #                 augmentation=augmentation)
    #
    # elif args.command == "evaluate":
    #     # Validation dataset
    #     dataset_val = AbnormalityDataset()
    #     val_type = "val" if args.year in '2017' else "minival"
    #     coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
    #     dataset_val.prepare()
    #     print("Running COCO evaluation on {} images.".format(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'evaluate'".format(args.command))

