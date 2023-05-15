import numpy as np
import cv2 as cv
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# general settings for plots
matplotlib.rcParams.update({'font.size': 14})
sns.set_style('whitegrid')
sns.set_palette('colorblind')
sns.set_context('paper', font_scale=2.0)


# class definitions
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# define class for storing image data with methods for watershed segmentation, addition of bounding ellipses and saving of results
class Image:
    def __init__(self, filename):
        self.filename = filename
        self.img = cv.imread(filename)
        self.labels = None
        self.cleaned_labels = None
        self.area = None
        self.axes = None
        self.img_with_ellipses = None
    
    # preprocess image and apply watershed algorithm to detect, detach and label cells
    def apply_watershed(self):
        """ Preprocess image and apply watershed algorithm to detect, detach and label cells
        parameters
        ----------
        self.img: source image
        """
        # convert to grayscale
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # apply Gaussian blur to remove noise
        gray = cv.GaussianBlur(gray, (5,5), 0)

        # binarize image using Otsu's method
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # noise removal by morphological opening
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # identify 'sure background' area by dilation
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # identify 'sure foreground' area by distance transform and thresholding
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        th = dist_transform.mean() + dist_transform.std()
        _, sure_fg = cv.threshold(dist_transform, th, 255, 0)

        # find unknown region by subtracting 'sure foreground' from 'sure background'
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # identify foreground objects
        _, markers = cv.connectedComponents(sure_fg)

        # add one to all labels so that 'sure background' is not 0, but 1, as required by watershed algorithm
        markers = markers + 1

        # mark unknown region as 0 as required by watershed algorithm
        markers[unknown==255] = 0

        # apply watershed algorithm to assign unknown regions to foreground, background or boundary
        self.labels = cv.watershed(self.img, markers)
        return self.labels
    
    # find enclosing ellipses using cv.minAreaRect and add to image
    def add_enclosing_ellipses(self):
        """ Find enclosing ellipse using cv.minAreaRect
        parameters
        ----------
        self.img: source image
        self.labels: destination labeled image
        """
        img_copy = self.img.copy()
        labels_copy = self.labels.copy()
        d = []
        area = []

        # iterate over all objects
        for i in [x for x in np.unique(self.labels) if (x != -1 and x != 1)]:
            # find all points belonging to object
            points = np.where(self.labels == i)

            # discard if object is 
            # - too small
            # - too large
            # - at image border
            if len(points[0]) < 1000 \
                or np.subtract(points[0].max(), points[0].min()) > 500 \
                or np.subtract(points[1].max(), points[1].min()) > 500 \
                or points[0].min() == 1 or points[0].max() == self.labels.shape[0]-2 \
                or points[1].min() == 1 or points[1].max() == self.labels.shape[1]-2:
                labels_copy[labels_copy == i] = 1
                continue
            
            # find minimal bounding rectangle (possibly rotated)
            points = np.transpose(points)
            (xc, yc), (d1, d2), angle = cv.minAreaRect(points)

            # discard if area of ellipse is significantly bigger than area of object
            # indicating multiple cells being recognized as one
            a = np.pi*(d1/2)*(d2/2)
            if len(points) < 0.9*a:
                labels_copy[labels_copy == i] = 1
                continue

            # discard if ellipse is too far from circle
            # indicating an elongated assembly of cells being recognized as a single cell
            if min(d1, d2)/max(d1, d2) < 0.5:
                labels_copy[labels_copy == i] = 1
                continue
            
            # save diameters and area
            d.append((min(d1, d2), max(d1, d2)))
            area.append(a)

            # add enclosing ellipses around objects
            ellipse = ((int(yc), int(xc)), (int(d2), int(d1)), -angle)
            cv.ellipse(img_copy, ellipse, (0, 255, 0), 3)

        # store cleaned labels, image with ellipses, axes and area
        # axes and area are scaled to micrometers
        # 100 mikrometer: 94 pixel
        # 1 pixel: 100/94 mikrometer
        # area(pixel) = (100/94)^2 mikrometer^2
        self.axes = np.array(d) * (100/94)
        self.area = np.array(area) * ((100/94)**2)
        self.cleaned_labels = labels_copy
        self.img_with_ellipses = img_copy
        return self.img_with_ellipses, self.cleaned_labels, self.area, self.axes

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# function definitions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# split lists containing area and diameters of objects into three lists according to suffix of filename 
# i.e., 'yy', 'yn', 'nn', that is, according to annotation
def split_by_annotation(area, diams, filenames):
    """ Split area and diams into three lists according to suffix of filename (i.e., 'yy', 'yn', 'nn')
    parameters
    ----------
    area: list of areas
    diams: list of diameters
    filenames: list of filenames
    """
    area_yy = []
    area_yn = []
    area_nn = []
    diams_yy = []
    diams_yn = []
    diams_nn = []

    for i in range(len(filenames)):
        if filenames[i].split('.')[0][-2:] == 'yy':
            area_yy.append(area[i])
            diams_yy.append(diams[i])
        elif filenames[i].split('.')[0][-2:] == 'yn':
            area_yn.append(area[i])
            diams_yn.append(diams[i])
        elif filenames[i].split('.')[0][-2:] == 'nn':
            area_nn.append(area[i])
            diams_nn.append(diams[i])
    return area_yy, area_yn, area_nn, diams_yy, diams_yn, diams_nn


# create kde plot of cell sizes grouped by annotation
def kde_plot_area(area_yy, area_yn, area_nn):
    # create figure
    fig, ax = plt.subplots()
    sns.kdeplot(np.concatenate(area_yy), ax=ax, fill=True, label='yy')
    sns.kdeplot(np.concatenate(area_yn), ax=ax, fill=True, label='yn')
    sns.kdeplot(np.concatenate(area_nn), ax=ax, fill=True, label='nn')
    plt.xlabel('Area in ' + r'$\mu$' + 'm' + r'$^2$')
    plt.ylabel('Density')
    plt.legend()

    # change x-axis format to scientific notation
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()


# create kde plot of cell sizes overlaid on histograms grouped by annotation
def kde_hist_plot_area(area_yy, area_yn, area_nn):
    # create figure
    fig, ax = plt.subplots()
    sns.histplot(np.concatenate(area_yy), ax=ax, label='yy', kde=True, binwidth=5000, stat='density')
    sns.histplot(np.concatenate(area_yn), ax=ax, label='yn', kde=True, binwidth=5000, stat='density')
    sns.histplot(np.concatenate(area_nn), ax=ax, label='nn', kde=True, binwidth=5000, stat='density')
    ax.set_xlabel('Area (' + r'$\mu$' + 'm' + r'$^2$' + ')')
    ax.set_ylabel('Density')
    ax.legend()

    # change x-axis format to scientific notation
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    # path to folder containing images
    path = 'base_model_pictures'

    # list of all filenames in specified folder
    filenames = os.listdir(path)

    results = []
    results_cleaned = []
    diams = []
    size = []

    # iterate over all files
    for filename in filenames:
        # path to image
        file = os.path.join(path, filename)
        print(file)

        # read image
        img = Image(file)

        # apply watershed algorithm
        labels = img.apply_watershed()
        # plt.imsave('base_model_watershed/' + filename.split('.')[0] + '_wtrshd.tiff', labels)
        results.append(labels)

        # fit enclosing ellipses and add to image
        img_with_ellipses, labels_cleaned, a, d = img.add_enclosing_ellipses()
        # cv.imwrite('base_model_ellipses/' + filename.split('.')[0] + '_elps.tif', img_with_ellipses)

        # save axes lengths, areas and cleaned labels
        diams.append(d)
        size.append(a)
        results_cleaned.append(labels_cleaned)


    # split area and diams into three lists according to suffix of filename (i.e., 'yy', 'yn', 'nn')
    # that is, according to annotation
    area_yy, area_yn, area_nn, diams_yy, diams_yn, diams_nn = split_by_annotation(size, diams, filenames)

    # plot kde plots of area for yy, yn and nn
    kde_plot_area(area_yy, area_yn, area_nn)

    # plot kde plots of area for yy, yn and nn overlaid on histograms
    kde_hist_plot_area(area_yy, area_yn, area_nn)


if __name__ == '__main__':
    main()