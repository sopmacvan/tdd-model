import os
import json
from shapely.geometry import Polygon
from shapely.ops import unary_union
import skimage.draw
import random

random.seed(69)


def get_per_image_annot(dataset_dir):
    def sort_polygons_by_centroid(polygons):
        # Calculate centroids and sort polygons by x-coordinate of centroids
        centroids = [polygon.centroid.x for polygon in polygons]
        sorted_indices = sorted(range(len(centroids)), key=lambda i: centroids[i])
        return [polygons[i] for i in sorted_indices]

    image_dir = os.path.join(dataset_dir, "Tufts Dental Database\\Radiographs")
    annot_dir = os.path.join(dataset_dir, "Tufts Dental Database\\Expert\\expert.json")

    per_image_annot = []

    annotations = json.load(open(annot_dir))
    for annot in annotations:
        # get external id, polygons, class
        name_dict = {
            "none": 0,
            "benign_cyst_neoplasia": 1,
            "malignant_neoplasia": 2,
            "inflammation": 3,
            "dysplasia": 4,
            "metabolic/systemic": 5,
            "trauma": 6,
            "developmental": 7,
        }
        chars = []
        polygons = []
        num_ids = []

        description = annot['Description']
        external_id = annot['External ID']
        for obj in annot['Label']['objects']:
            if obj['polygons'] == 'none':
                break

            level_one = obj['value']
            level_two = ''
            level_three = []
            level_four = []
            categories = []

            # get radiographic characteristics and category labels
            for c in obj['classifications']:
                level = c['value']

                if level == 'level_one':
                    level_two = c['answer']['value']
                elif level == 'level_two':
                    level_three = c['answer']['value']
                elif level == 'level_three':
                    if 'answer' in c:
                        level_four.append(c['answer']['value'])
                    elif 'answers' in c and not level_three:
                        for ans in c['answers']:
                            level_four.append(ans['value'])
                elif level == 'level_four':
                    if 'answer' in c:
                        categories.append(c['answer']['value'])
                    elif 'answers' in c and not categories:
                        for ans in c['answers']:
                            categories.append(ans['value'])

            # get polygons per abnormality
            last_appended_polygon = None  # Track the last appended polygon
            distance_threshold = 10  # Define a distance threshold for merging polygons

            preprocessed_polygons = obj['polygons']
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
                    all_levels = [level_one, level_two, level_three, level_four]
                    chars.append(all_levels)
                    polygons.append([list(p.exterior.coords)])
                    # num_ids.append([name_dict[category] for category in categories])
                    num_ids.append(name_dict[categories[0]])
                    last_appended_polygon = p

        image_path = os.path.join(image_dir, external_id)
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        entry = {
            'image_id': external_id,
            'path': image_path,
            'width': width,
            'height': height,
            'num_ids': num_ids,
            'polygons': polygons,
            'chars': chars,
            'description': description,
        }

        per_image_annot.append(entry)
    return per_image_annot


def create_train_val_split(dataset_dir):
    annotations = get_per_image_annot(dataset_dir)

    # Calculate class counts
    class_counts = {}  # Count occurrences of each class in the dataset
    for entry in annotations:
        for class_id in entry['num_ids']:
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

    # Calculate class proportions for each class
    desired_proportions = {}  # Define desired proportions for each class
    for class_id, count in class_counts.items():
        desired_proportions[class_id] = count * 0.8  # 80% for training

    # Initialize training and validation sets
    train_set = []
    val_set = []

    # Create a copy of class_counts to track class counts during the split
    class_counts_copy = class_counts.copy()

    # Define a function to check if the desired proportion is met for a class
    def is_desired_proportion_met(class_id):
        return class_counts_copy[class_id] >= desired_proportions[class_id]

    # Shuffle the data to ensure randomness
    random.shuffle(annotations)

    # Split the data into training and validation sets while maintaining desired proportions
    for entry in annotations:
        if not entry['num_ids']:
            # Handle images with no labeled objects by distributing them evenly between train and val
            if len(train_set) < len(val_set):
                train_set.append(entry)
            else:
                val_set.append(entry)
        else:
            # Check if the desired proportion is met for each class in the entry
            should_be_in_train = all(is_desired_proportion_met(class_id) for class_id in entry['num_ids'])

            if should_be_in_train:
                train_set.append(entry)
                for class_id in entry['num_ids']:
                    class_counts_copy[class_id] -= 1
            else:
                val_set.append(entry)

    train_set, val_set = val_set, train_set

    # Save the training set as a separate JSON file
    with open(os.path.join(dataset_dir, 'train.json'), 'w') as train_json_file:
        json.dump(train_set, train_json_file, indent=4)

    # Save the validation set as a separate JSON file
    with open(os.path.join(dataset_dir, 'val.json'), 'w') as val_json_file:
        json.dump(val_set, val_json_file, indent=4)


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("../")
    dataset_dir = os.path.join(ROOT_DIR, 'dataset')
    create_train_val_split(dataset_dir)
