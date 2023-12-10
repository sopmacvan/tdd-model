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
from scipy.sparse import coo_matrix


def are_polygons_close(polygon1, polygon2, distance_threshold):
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    distance = poly1.boundary.distance(poly2.boundary)
    return distance <= distance_threshold

def group_polygons_by_outline(polygons, distance_threshold):
    groups = []
    visited_indices = set()

    for i, polygon in enumerate(polygons):
        if i not in visited_indices:
            group = [polygons[j] for j in range(len(polygons)) if are_polygons_close(polygon, polygons[j], distance_threshold)]
            groups.append(group)
            visited_indices.update(j for j in range(len(polygons)) if are_polygons_close(polygon, polygons[j], distance_threshold))

    return groups

def get_abnormality_dicts(dataset_dir):
    "get abnormalities information in the standard dict format"
    image_dir = os.path.join(dataset_dir, "Tufts Dental Database/Radiographs")
    annot_dir = os.path.join(dataset_dir, "Tufts Dental Database/Expert/expert.json")

    dataset_dicts = []

    img_anns = json.load(open(annot_dir))
    for _, v in tqdm(enumerate(img_anns), total=len(img_anns)):
        record = {}

        file_name = os.path.join(image_dir, v['External ID'])
        height, width = cv2.imread(file_name).shape[:2]
        
        record['file_name'] = file_name
        record['height'] = height
        record['width'] = width
        record['annotations'] = []
        record['class_counts'] = dict()


        polygons = []
        categories = []

        name_dict = {
            "benign_cyst_neoplasia": 1,
            "developmental": 2,
            "inflammation": 3,
            
        }

        available_categories = name_dict.keys()
        for c in available_categories:
            record['class_counts'][c] = 0

        annos = v['Label']['objects']  
        for anno in annos:
            # get categories 
            anno_c = anno['classifications']
            if anno_c == 'none':
                continue

            obj_category = None
            for c in anno_c:
                if c['value'] == 'level_four':
                    if 'answer' in c:
                        value = c['answer']['value']
                        if value in available_categories:
                            obj_category = value  
                            break
                    elif 'answers' in c:
                        multi_values = {multi_c['value'] for multi_c in c['answers']}
                        for value in available_categories:
                            if value in multi_values:
                                obj_category = value
                                break

            # get polygons
            if not obj_category:
                continue
            
            obj_polygons = [p for p in anno['polygons'] if len(p) > 2]
            
            distance_threshold = 20  
            grouped_polygons = group_polygons_by_outline(obj_polygons, distance_threshold)

            categories.append(obj_category)
            polygons.append(grouped_polygons)

        # get annotations for each object in an image
        mask = np.zeros([height, width], dtype=np.uint8)

        for obj_category, grouped_polygons in zip(categories, polygons):
            for obj_polygons in grouped_polygons:
                # convert polygons to binary mask
                for p in obj_polygons:
                    p_mask = skimage.draw.polygon2mask(image_shape=(height, width),
                                                        polygon=[[y, x] for x, y in p]).astype(np.uint8)
                    mask[p_mask==1] = name_dict[obj_category]

                record['class_counts'][obj_category] = record['class_counts'].get(obj_category, 0) + len(obj_polygons) 
        # convert mask to sparse mask
        sparse_mask = coo_matrix(mask)
        record['annotations'] = sparse_mask
        dataset_dicts.append(record)
        
    return dataset_dicts


def get_class_counts(dataset_dicts):
  "count the objects in each class of a dataset dict"
  class_counts = dict()

  # Iterate over each image in the dataset
  for image in dataset_dicts:
      img_class_counts = image['class_counts']
      for category in img_class_counts:
          class_counts[category] = class_counts.get(category, 0) + img_class_counts[category]

  return class_counts


def train_test_split_by_objects(dataset_dicts, test_size=.2, random_seed=69):
  "create train test split by object classes"
  data = dataset_dicts.copy()
  random.seed(random_seed)
  random.shuffle(data)

  class_counts = get_class_counts(data)

  total_test_images = test_size * len(data)
  test_class_counts = {}
  
  for category, count in class_counts.items():
      test_class_counts[category] = int(count * test_size)

  # Define lists to store the images for each set
  train_data = []
  test_data = []

  # Iterate over each image in the dataset
  for image in data:
      # Get the number of objects for each class in this image
      img_class_counts = image['class_counts']

      # when image has no objects
      if not img_class_counts:
          if len(test_data) < total_test_images:
              test_data.append(image)
          else:
              train_data.append(image)
      # when image has objects
      else:
          if any(test_class_counts[category] > 0 for category in img_class_counts if category in test_class_counts):
              for category in img_class_counts:
                  test_class_counts[category] = test_class_counts.get(category, 0) - img_class_counts[category]
              test_data.append(image)
          else:
              train_data.append(image)

  return train_data, test_data

def oversample_minor_class(dataset_dicts, target_ratio=.8, random_seed=69):
    "oversample minor class"
    random.seed(random_seed)

    data = dataset_dicts.copy()
    class_counts = get_class_counts(data)
    major_class = max(class_counts, key=class_counts.get)
    minor_classes = [c for c in class_counts if c != major_class]
    
    imgs_with_minor_classes = {c:[] for c in minor_classes}
  
    # isolate images with minor classes
    for image in data:
        img_class_counts = image['class_counts']
        for c in imgs_with_minor_classes:
            if c in img_class_counts and img_class_counts[c] > 0:
                imgs_with_minor_classes[c].append(image)
    
    # oversample minor classes
    target = int(class_counts[major_class] * target_ratio)
    
    print(f'BEFORE oversampling \t len {len(data)}, class_counts {class_counts}')
    for c in minor_classes:
        curr_class_counts = get_class_counts(data)
        while curr_class_counts[c] < target:
            imgs_to_add = random.choices(imgs_with_minor_classes[c], k= 10)
            data += imgs_to_add
            curr_class_counts = get_class_counts(data)
    
    print(f'AFTER oversampling \t len {len(data)}, class_counts {get_class_counts(data)}')
    return data


def save_dataset(path, data):
    "save dataset as .pkl file"
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def load_dataset(path):
    "load .pkl file dataset"
    with open(path, 'rb') as f:
      data = pickle.load(f)
    return data

def create_train_val_test_dataset(dataset_path):
    "create dataset"
    train_path = os.path.join(dataset_path, 'train.pkl')
    val_path = os.path.join(dataset_path, 'val.pkl')
    test_path = os.path.join(dataset_path, 'test.pkl')

    # load datasets if they exist
    if os.path.exists(test_path):
        train = load_dataset(train_path)
        val = load_dataset(val_path)
        test = load_dataset(test_path)
    # create datasets
    else:
        dataset_dicts = get_abnormality_dicts(dataset_path)
        train, val = train_test_split_by_objects(dataset_dicts, test_size=.2)
        val, test = train_test_split_by_objects(val, test_size=.5)
        
        save_dataset(train_path, train)
        save_dataset(val_path, val)
        save_dataset(test_path, test)
    
    return train, val, test

if __name__ == '__main__':
    dataset_path = '/content/drive/MyDrive/Github/tdd-model/dataset'
    train, val, test = create_train_val_test_dataset(dataset_path)
    print(f"train: \t len {len(train)}, class_counts {get_class_counts(train)}")
    print(f"val: \t len {len(val)}, class_counts {get_class_counts(val)}")
    print(f"test: \t len {len(test)}, class_counts {get_class_counts(test)}")

    # perform oversampling on train set
    train = oversample_minor_class(train)
    # save oversampled train set
    train_path = os.path.join(dataset_path, 'train.pkl')
    save_dataset(train_path, train)

    print(f"Successfully created datasets on {dataset_path}")
