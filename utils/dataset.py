import os
import json
import pickle
from shapely.geometry import Polygon
from shapely.ops import unary_union
import skimage.draw
import cv2
import random
from pycocotools import mask as maskUtils
from tqdm import tqdm
import numpy as np

def get_abnormality_dicts(dataset_dir, bbox_mode):
    def sort_polygons_by_centroid(polygons):
        # Calculate centroids and sort polygons by x-coordinate of centroids
        centroids = [polygon.centroid.x for polygon in polygons]
        sorted_indices = sorted(range(len(centroids)), key=lambda i: centroids[i])
        return [polygons[i] for i in sorted_indices]

    image_dir = os.path.join(dataset_dir, "Tufts Dental Database/Radiographs")
    annot_dir = os.path.join(dataset_dir, "Tufts Dental Database/Expert/expert.json")

    dataset_dicts = []

    img_anns = json.load(open(annot_dir))
    for idx, v in tqdm(enumerate(img_anns), total=len(img_anns)):
        record = {}

        file_name = os.path.join(image_dir, v['External ID'])
        height, width = cv2.imread(file_name).shape[:2]

        record['file_name'] = file_name
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        record['annotations'] = []

        name_dict = {
            "benign_cyst_neoplasia": 0,
            "malignant_neoplasia": 1,
            "inflammation": 2,
            "dysplasia": 3,
            "metabolic/systemic": 4,
            "trauma": 5,
            "developmental": 6,
            "none": -1
        }
        
        category_ids = []
        polygons = []

        objs = []
        annos = v['Label']['objects']

        # get category_id and segmentations
        for anno in annos:
            if anno['polygons'] == 'none':
                break

            # level_one = obj['value']
            # level_two = ''
            # level_three = []
            # level_four = []
            # chars = []
            
            # get categories
            categories = []
            for c in anno['classifications']:
                level = c['value']

                # if level == 'level_one':
                #     level_two = c['answer']['value']
                # elif level == 'level_two':
                #     level_three = c['answer']['value']
                # elif level == 'level_three':
                #     if 'answer' in c:
                #         level_four.append(c['answer']['value'])
                #     elif 'answers' in c and not level_three:
                #         for ans in c['answers']:
                #             level_four.append(ans['value'])
                if level == 'level_four':
                    if 'answer' in c:
                        categories.append(c['answer']['value'])
                    elif 'answers' in c and not categories:
                        for ans in c['answers']:
                          categories.append(ans['value'])

            # get polygons
            last_appended_polygon = None  # Track the last appended polygon
            distance_threshold = 10  # Define a distance threshold for merging polygons

            preprocessed_polygons = anno['polygons']
            preprocessed_polygons = [Polygon(p) for p in preprocessed_polygons if
                                     len(p) > 2]  # get only polygons with enough vertices
            preprocessed_polygons = sort_polygons_by_centroid(
                preprocessed_polygons)  # sort by how near to neighbor polygon

            for p in preprocessed_polygons:
                if last_appended_polygon is not None and p.distance(
                        last_appended_polygon) < distance_threshold:
                    last_appended_polygon = unary_union([last_appended_polygon, p])
                    polygons[-1].append(list(p.exterior.coords))

                else:
                    # all_levels = [level_one, level_two, level_three, level_four]
                    # chars.append(all_levels)
                    polygons.append([list(p.exterior.coords)])
                    # category_ids.append([name_dict[category] for category in categories])
                    category_ids.append(name_dict[categories[0]])
                    last_appended_polygon = p

        # get annotations for each object in an image
        for category_id, object_polygons in zip(category_ids, polygons):
            # convert polygons to binary mask
            mask = np.zeros([height, width],
                          dtype=np.uint8)
            for p in object_polygons:
                p_mask = skimage.draw.polygon2mask(image_shape=(height, width),
                                                    polygon=[[y, x] for x, y in p]).astype(np.uint8)
                mask = np.logical_or(mask, p_mask)

            # compute the bbox coordinates from binary mask
            indices = np.where(mask)

            x_min = np.min(indices[1])
            y_min = np.min(indices[0])
            x_max = np.max(indices[1])
            y_max = np.max(indices[0])

            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            # convert binary mask to RLE-encoded mask
            rle = maskUtils.encode(np.asfortranarray(mask))
            
            instance_annotation = {
                'bbox': bbox,
                'bbox_mode': bbox_mode,
                'category_id': category_id,
                'segmentation': rle
            }

            if instance_annotation['category_id'] == -1:
              continue
            record['annotations'].append(instance_annotation)
        dataset_dicts.append(record)

    return dataset_dicts


def get_class_counts(dataset_dicts):
  class_counts = {}

  # Iterate over each image in the dataset
  for image in dataset_dicts:
      # Iterate over each annotation in the image
      for annotation in image['annotations']:
          # Get the category ID for the annotation
          category_id = annotation['category_id']
          
          # Increment the count for this category ID
          class_counts[category_id] = class_counts.get(category_id, 0) + 1

  class_counts = dict(sorted(class_counts.items()))
  return class_counts


def create_train_val_split(dataset_dicts, train_path, val_path):
  data = dataset_dicts.copy()
  random.seed(69)
  random.shuffle(data)

  class_counts = get_class_counts(data)

  # Split each class into train-val 70:30 ratio
  total_val_images = 0.3 * len(data)
  val_class_counts = {}
  
  for category_id, count in class_counts.items():
      val_class_counts[category_id] = int(count * 0.3)

  # Define lists to store the images for each set
  train_data = []
  val_data = []

  # Iterate over each image in the dataset
  for image in data:
      # Get the number of objects for each class in this image
      object_counts = {}
      for annotation in image['annotations']:
          category_id = annotation['category_id']
          object_counts[category_id] = object_counts.get(category_id, 0) + 1
      
      # when image has no objects
      if not object_counts:
        if len(val_data) < total_val_images:
          val_data.append(image)
        else:
          train_data.append(image)
      # when image has objects
      else:
        if any(val_class_counts[key] > 0 for key in object_counts if key in val_class_counts):
          for key in val_class_counts:
              if key in object_counts and val_class_counts[key] > 0:
                  val_class_counts[key] -= object_counts[key]
          val_data.append(image)
        else:
          train_data.append(image)


  # Save the training and validation datasets as separate JSON files
  with open(train_path, 'wb') as f:
      pickle.dump(train_data, f)

  with open(val_path, 'wb') as f:
      pickle.dump(val_data, f)

  return train_data, val_data



