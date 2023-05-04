import numpy as np
import cv2 as cv
import napari


def visualize_napari(images, labels):
    """
    Visualize all images in list using napari.
    ----------
    images: list of numpy images
    labels: list of names for each image
    """
    viewer = napari.Viewer()
    with napari.gui_qt():
        for i, img in enumerate(images):
            viewer.add_image(img, name=labels[i])


def clean_stats(labels, stats, centroids):
    """
    Delete connected components that are too small or too large.
    ----------
    labels: destination labeled image
    stats: topmost coordinate, leftmost coordinate, width, height and area of connected components
    centroids: coordinates of centroids of connected components
    """
    # create copies
    labels_cp = labels.copy()
    stats_cp = stats.copy()
    centroids_cp = centroids.copy()
    indices = []

    for i in range(0, stats.shape[0]):
        # discard object if too small
        if stats[i, 4] < 100:
            indices.append(i)
        # discard object if too big
        elif stats[i, 4] > np.pi*400**2:
            indices.append(i)
        # discard object if ratio of width to height not within specified tolerance
        elif not(0.7 <= stats[i,2] / stats[i,3] <= 1.3):
            indices.append(i)
    
    # remove corresponding entries from lables, stats and centroids
    labels_cp = labels_cp * ~(np.in1d(labels_cp, indices).reshape(labels_cp.shape))
    stats_cp = np.delete(stats_cp, indices, axis=0)
    centroids_cp = np.delete(centroids_cp, indices, axis=0)
    return labels_cp, stats_cp, centroids_cp


def find_enclosing_ellipses(img, labels):
    """
    Find and add enclosing ellipses around objects
    ----------
    labels: contains only labels of "valid" components
    """
    img_copy = img.copy()
    stats = []

    for i in np.unique(labels)[1:]:
        # set of points of object
        points = np.where(labels == i)
        points = np.transpose(points)

        # find minimal bounding rectangle, possibly rotated
        (xc, yc), (d1, d2), angle = cv.minAreaRect(points)

        # save data
        stats.append(((xc, yc), (d1, d2), angle))

        # add bounding ellipse around objects
        ellipse = ((int(yc), int(xc)), (int(d2), int(d1)), -angle)
        cv.ellipse(img_copy, ellipse, (255, 255, 0), 3)
    return img_copy, stats


def add_bounding_boxes(img, stats, labels, centroids):
    """
    Add rectangles around objects
    """
    labels_cp = labels.copy()
    stats_cp = stats.copy()
    centroids_cp = centroids.copy()
    indices = []
    img_copy = img.copy()
    
    for i in range(0, stats.shape[0]):
        # discard component if too small indicating noise or holes within components
        if stats[i, 4] < 200:
            indices.append(i)
            continue
        # color box red if component too big indicating two components being attached to each other
        # or if ratio between width and height of component outside tolerance 
        # indicating two components being attached to each other or component being at boundary and thus cropped
        elif stats[i, 4] > np.pi*400**2 or not(0.7 <= stats[i,2] / stats[i,3] <= 1.3):
            indices.append(i)
            x = stats[i, 0] - 10
            y = stats[i, 1] - 10
            w = stats[i, 2] + 20
            h = stats[i, 3] + 20
            cv.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # else, color box green
        else:
            x = stats[i, 0] - 10
            y = stats[i, 1] - 10
            w = stats[i, 2] + 20
            h = stats[i, 3] + 20
            cv.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    labels_cp = labels_cp * ~(np.in1d(labels_cp, indices).reshape(labels_cp.shape))
    stats_cp = np.delete(stats_cp, indices, axis=0)
    centroids_cp = np.delete(centroids_cp, indices, axis=0)
    return img_copy, labels_cp, stats_cp, centroids_cp


# def add_bounding_boxes(img, stats):
#     """
#     Adds enclosing rectangles around objects
#     ----------
#     img: destination image
#     stats: topmost coordinate, leftmost coordinate, width, height and area of connected components
#     """
#     img_copy = img.copy()
#     for i in range(1, len(stats)):
#         x = stats[i, 0] - 10
#         y = stats[i, 1] - 10
#         w = stats[i, 2] + 20
#         h = stats[i, 3] + 20
#         cv.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     return img_copy


paths = ['DE_annotated_pictures/20220918_809 D4001_ch00_yy.tif',
         'DE_annotated_pictures/20220918_811 D4001_ch00_yn.tif',
         'DE_annotated_pictures/20220130_543_D4001_ch00_nn.tif']

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))

for path in paths:
    # read image
    img = cv.imread(path)

    # convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # binarize image applying global thresholding
    _, thresh = cv.threshold(gray, np.mean(gray), 255, cv.THRESH_BINARY_INV)

    # erode boundary to detach objects
    erosion = cv.erode(thresh, kernel, iterations = 5)

    # compute connected components and corresponding stats
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(erosion, connectivity=8)

    # add bounding boxes around objects and correct stats for neglected objects
    img_with_boxes, tlabels, tstats, tcentroids = add_bounding_boxes(img, stats, labels, centroids)

    # add enclosing ellipses
    img_with_ellipses, ellipse_data = find_enclosing_ellipses(img, tlabels)

    images = [img, gray, thresh, erosion, img_with_boxes, img_with_ellipses]
    labels = ['img', 'gray', 'thresh', 'erosion', 'img_with_boxes', 'img_with_ellipses']
    visualize_napari(images, labels)
