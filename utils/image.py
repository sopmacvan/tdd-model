from PIL import Image, ImageDraw


def draw_polygon_mask(color, polygons, image):
    """
    draw polygon mask on an image

    :param color: str
    :param polygons: list of list
    :param image: Image
    :return image: Image
    """

    def no_polygons():
        """check if there are polygons"""
        return type(polygons) != list

    if no_polygons():
        return image

    for p in polygons:
        xy_coordinates = list(map(tuple, p))
        abnormality_mask = image.copy()
        draw = ImageDraw.Draw(abnormality_mask)
        try:
            draw.polygon(xy_coordinates, fill=color, outline=color)
        except TypeError:
            continue

        image = Image.blend(image, abnormality_mask, 0.5)
    return image
