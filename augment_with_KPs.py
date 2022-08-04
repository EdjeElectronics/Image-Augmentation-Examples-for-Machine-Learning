#### Image Augmentation for Object Detection Models
#
# Author: Evan Juras
# Date: 5/15/20
# Description:
# This program takes a set of original images and creates augmented images that can be used
# as additional training data for an object detection model. It loads each original image,
# creates N randomly augmented images, and saves the new images and annotation data. It keeps
# track of bounding boxes that are needed for training object detection models.

import imgaug as ia
from imgaug import augmenters as iaa
import os
import sys
import argparse
from glob import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np


#### Define control variables and parse user inputs
parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', help='Folder containing images to augment',
                    required=True)
parser.add_argument('--imgext', help='File extension of images (for example, .JPG)',
                    default='.JPG')
parser.add_argument('--labels', help='Text file with list of classes',
                    required=True)
parser.add_argument('--numaugs', help='Number of augmented images to create from each original image',
                    default=5)
parser.add_argument('--debug', help='Displays every augmented image when enabled',
                    default=False)

args = parser.parse_args()
IMG_DIR = args.imgdir
if not os.path.isdir(IMG_DIR):
    print('%s is not a valid directory.' % IMG_DIR)
    sys.exit(1)
IMG_EXTENSION = args.imgext
LABEL_FN = args.labels
if not os.path.isfile(LABEL_FN):
    print('%s is not a valid file path.' % LABEL_FN)
    sys.exit(1)
NUM_AUG_IMAGES = int(args.numaugs)
debug = args.debug
cwd = os.getcwd()


#### Define augmentation sequence ####
# This can be tweaked to create a huge variety of image augmentations.
# See https://github.com/aleju/imgaug for a list of augmentation techniques available.
seq1 = iaa.Sequential([
    iaa.Fliplr(0.5),                             # Horizontal flip 50% of images
    iaa.Crop(percent=(0, 0.20)),                 # Crop all images between 0% to 20%
    #iaa.GaussianBlur(sigma=(0, 1)),             # Add slight blur to images
    #iaa.Multiply((0.7, 1.3), per_channel=0.2)), # Slightly brighten, darken, or recolor images
##    iaa.Affine(
##        scale={"x": (0.8, 1.2), "y": (0.8,1.2)},                # Resize image
##        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # Translate image
##        rotate=(-5, 5),                                         # Rotate image
##        mode=ia.ALL, cval=(0,255)                               # Filling in extra pixels
##        )
    ])

#### Function definitions ####
# Function for reading annotation data from a XML file
def read_annotation_data(xml_fn):
    file = open(xml_fn,'r')
    tree = ET.parse(file)
    root = tree.getroot()
    size = root.find('size')
    imw = int(size.find('width').text)
    imh = int(size.find('height').text)
    objects = []
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)
        bb = [(xmin,ymin),(xmax,ymin),(xmax, ymax),(xmin,ymax)] # Top left, top right, bottom right, bottom left
        objects.append([cls,bb])

    return imw, imh, objects
    

# Function for finding bounding box from keypoints
def kps_to_BB(kps,imgW,imgH):
    """
        Determine imgaug bounding box from imgaug keypoints
    """
    extend=1 # To make the bounding box a little bit bigger
    kpsx=[kp[0] for kp in kps]
    xmin=max(0,int(min(kpsx)-extend))
    xmax=min(imgW,int(max(kpsx)+extend))
    kpsy=[kp[1] for kp in kps]
    ymin=max(0,int(min(kpsy)-extend))
    ymax=min(imgH,int(max(kpsy)+extend))
    if xmin==xmax or ymin==ymax:
        return None
    else:
        #return ia.BoundingBox(x1=xmin,y1=ymin,x2=xmax,y2=ymax)
        return [(xmin, ymin),(xmax, ymax)]

# Define XML annotation format for creating new XML files
xml_body_1="""<annotation>
        <folder>{FOLDER}</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""

# Function to create XML files
def create_xml(folder, image_path, xml_path, size, imBBs):

    # Get image size and filename
    imH, imW = size[0], size[1]
    image_fn = os.path.split(image_path)[-1]

    # Create XML file and write data
    with open(xml_path,'w') as f:
        f.write(xml_body_1.format(**{'FOLDER':folder, 'FILENAME':image_fn, 'PATH':image_path,
                                     'WIDTH':imW, 'HEIGHT':imH}))

        for bbox in imBBs:
            f.write(xml_object.format(**{'CLASS':bbox[0], 'XMIN':bbox[1][0], 'YMIN':bbox[1][1],
                                          'XMAX':bbox[1][2], 'YMAX':bbox[1][3]}))

        f.write(xml_body_2)

    return


#### Start of main program ####

# Load classes from labelmap. Labelmap is a text file listing class names, with a newline between each class
with open(LABEL_FN,'r') as file:
    labels=file.read().split('\n')
classes = [label for label in labels if label!=''] # Remove blank labels (happens when there are extra lines at the end of the file)

# Obtain list of images in IMG_DIR directory
img_fns = glob(IMG_DIR + '/*' + IMG_EXTENSION)

# Go through every image in directory, augment it, and save new image/annotation data
for img_fn in img_fns:

    # Open image, get shape and base filename
    original_img = cv2.imread(img_fn)
    imgH, imgW, _ = original_img.shape
    base_img_fn = os.path.split(img_fn)[-1]
    base_fn = base_img_fn.replace(IMG_EXTENSION,'')

    # Read annotation data from image's corresponding XML file
    xml_fn = img_fn.replace(IMG_EXTENSION,'.xml')
    imgW_xml, imgH_xml, objects = read_annotation_data(xml_fn)
    if ((imgW_xml != imgW) or (imgH_xml != imgH)):
        print('Warning! Annotation data does not match image data for %s. Skipping image.' % img_fn)
        continue
    im_kps = []
    im_classes = []
    num_obj = len(objects)
    for obj in objects:
        im_classes.append(obj[0])
        im_kps.append(obj[1][0]) # Top left corner
        im_kps.append(obj[1][1]) # Top right corner
        im_kps.append(obj[1][2]) # Bottom right corner
        im_kps.append(obj[1][3]) # Bottom left corner

    # Define keypoints on image
    ia_kps = [ia.Keypoint(x=p[0], y=p[1]) for p in im_kps]
    original_kps = ia.KeypointsOnImage(ia_kps, shape=original_img.shape)

    # Define bounding boxes on original image (not needed for augmentation, but used for displaying later if debug == True)
    bboxes = []
    for i in range(num_obj):
        obj_kps = im_kps[i*4:(i*4+4)]
        bboxes.append(kps_to_BB(obj_kps,imgW,imgH))

    # Create new augmented images, and save them in folder with annotation data
    for n in range(NUM_AUG_IMAGES):

        # Define new filenames
        img_aug_fn = base_fn + ('_aug%d' % (n+1)) + IMG_EXTENSION
        img_aug_path = os.path.join(cwd,IMG_DIR,img_aug_fn)
        xml_aug_fn = img_aug_fn.replace(IMG_EXTENSION,'.xml')
        xml_aug_path = os.path.join(cwd,IMG_DIR,xml_aug_fn)

        # Augment image and keypoints.
        # First, copy original image and keypoints
        img = np.copy(original_img)
        kps = original_kps
        # Next, need to make sequence determinstic so it performs the same augmentation on image as it does on the keypoints
        seq1_det = seq1.to_deterministic()
        # Finally, run the image and keypoints through the augmentation sequence
        img_aug = seq1_det.augment_images([img])[0]
        kps_aug = seq1_det.augment_keypoints([kps])[0]
        imgH_aug, imgW_aug, _ = img_aug.shape

        # Extract augmented keypoints back into a list array, find BBs, and write annotation data to new file
        list_kps_aug = [(int(kp.x), int(kp.y)) for kp in kps_aug.keypoints]
        bboxes_aug = []
        bboxes_aug_data = []

        # Loop over every object, determine bounding boxes for new KPs, and save annotation data
        for i in range(num_obj):
            obj_aug_kps = list_kps_aug[i*4:(i*4+4)] # Augmented keypoints for each object
            obj_bb = kps_to_BB(obj_aug_kps,imgW_aug,imgH_aug) # Augmented bounding boxes for each object
            if obj_bb: # Sometimes the bbox coordinates are invalid and obj_bb is empty, so need to check if obj_bb valid
                bboxes_aug.append(obj_bb) # List of bounding boxes for each object
                xmin = int(obj_bb[0][0])
                ymin = int(obj_bb[0][1])
                xmax = int(obj_bb[1][0])
                ymax = int(obj_bb[1][1])
                coords = [xmin, ymin, xmax, ymax]
                label = im_classes[i]
                bboxes_aug_data.append([label, coords]) # List of bounding box data for each object (class name and box coordinates)

        # Save image and XML files to hard disk
        cv2.imwrite(img_aug_path,img_aug)
        create_xml(IMG_DIR, img_aug_fn, xml_aug_path, [imgH_aug,imgW_aug], bboxes_aug_data)

        # Display original and augmented images with keypoints (if debug is enabled)
        if debug:

            img_show = np.copy(img)
            img_aug_show = np.copy(img_aug)

            # Draw keypoints and BBs on normal image
            for bb in bboxes:
                cv2.rectangle(img_show, bb[0], bb[1], (50,255,50), 5)
            for kp in im_kps:
                cv2.circle(img_show,kp, 8, (255,255,0), -1)
                cv2.circle(img_show,kp, 9, (10,10,10), 2)

            # Draw keypoints and BBs on augmented image
            for bb in bboxes_aug:
                cv2.rectangle(img_aug_show, bb[0], bb[1], (50,255,50), 5)
            for kp in list_kps_aug:
                cv2.circle(img_aug_show,kp, 10, (255,255,0), -1)
                cv2.circle(img_aug_show,kp, 10, (10,10,10), 2)

            if n == 0:
                cv2.imshow('Normal',img_show)
            cv2.imshow('Augmented %d' % n,img_aug_show)
            cv2.waitKey()

# If images were displayed, close them at the end of the program
if debug:
    cv2.destroyAllWindows()
