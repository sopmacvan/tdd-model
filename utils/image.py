from PIL import Image, ImageDraw


def draw_polygon_mask(polygons, img_path):
    """
    draw polygon mask on an image

    :param polygons: list of list
    :param img_path: str
    :return image: Image
    """

    def no_polygons():
        """check if there are polygons"""
        return type(polygons) != list

    image = Image.open(img_path)

    if no_polygons():
        return image, 'no polygons found'

    for p in polygons:
        xy_coordinates = list(map(tuple, p))
        abnormality_mask = image.copy()
        draw = ImageDraw.Draw(abnormality_mask)
        try:
            draw.polygon(xy_coordinates, fill="#cc0000", outline="blue")
        except TypeError:
            continue

        image = Image.blend(image, abnormality_mask, 0.5)
    return image
