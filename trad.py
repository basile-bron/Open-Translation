import sys
import cv2

import os
import re
import math
import argparse
import numpy as np
import pytesseract
from googletrans import Translator
translator = Translator()
from PIL import Image, ImageFont, ImageDraw
import textwrap
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class Blurb(object):
    def __init__(self, x, y, w, h, text, confidence=100.0):
      """
      Initialize a Blurb object.

      Args:
          x (int): X-coordinate of the top-left corner of the region.
          y (int): Y-coordinate of the top-left corner of the region.
          w (int): Width of the region.
          h (int): Height of the region.
          text (str): The actual text content within the region.
          confidence (float, optional): Confidence level of the OCR system in recognizing the text content.
                                        Defaults to 100.0.
      """
      self.x = x
      self.y = y
      self.w = w
      self.h = h
      self.text = text
      self.confidence = confidence

    def clean_text(self):
      """
      Clean the text content by removing newline characters.

      Returns:
          str: The cleaned text.
      """
      text = self.text
      text = re.sub(r"\n", "", text)
      """
      for letter in text:

          if ord(letter) > 255:
              #print("deleting atrocity")
              text = text.replace(letter, "")
      """
      return text

    def __str__(self):
      """
      Return a string representation of the Blurb object.

      Returns:
          str: The string representation in the format: "x,y w x h confidence%: text".
      """
      return str(self.x) + ',' + str(self.y) + ' ' + str(self.w) + 'x' + str(self.h) + ' ' + str(self.confidence) + '% :' + self.text


class TranslatedBlurb(Blurb):
    def __init__(self, x, y, w, h, text, confidence, translation):
      """
      Initialize a TranslatedBlurb object.

      Args:
          x (int): X-coordinate of the top-left corner of the region.
          y (int): Y-coordinate of the top-left corner of the region.
          w (int): Width of the region.
          h (int): Height of the region.
          text (str): The actual text content within the region.
          confidence (float): Confidence level of the OCR system in recognizing the text content.
          translation (str): The translated text.
      """
      Blurb.__init__(self, x, y, w, h, text, confidence)
      self.translation = translation

    @classmethod
    def as_translated(cls, parent, translation):
      """
      Create a new TranslatedBlurb object based on an existing Blurb object and a translation.

      Args:
          parent (Blurb): The parent Blurb object to inherit attributes from.
          translation (str): The translated text.

      Returns:
          TranslatedBlurb: The new TranslatedBlurb object.
      """
      return cls(parent.x, parent.y, parent.w, parent.h, parent.text, parent.confidence, translation)

def translate_blurb(blurb, language):
    """
    Translate the text content of a Blurb object to English and create a TranslatedBlurb object.

      Args:
          blurb (Blurb): The Blurb object to be translated.

      Returns:
        blurb (Blurb): The Blurb object translated.
    """

    translated_text = translator.translate(blurb.clean_text(), dest=language)

    translation = translated_text.text.encode('utf-8', 'ignore')

    return TranslatedBlurb.as_translated(blurb, translation)



def flow_into_box(text, w, font, min_word_on_line=0.3):
    """
    Formats the given text to fit within a specified width.

    Args:
        text (str): The text to format.
        w (int): The width to fit the text into.
        font (PIL.ImageFont.FreeTypeFont, optional): The font to use for measuring text width. Defaults to None.
        min_word_on_line (float, optional): The minimum proportion of a word to be placed on a line. Defaults to 0.3.

    Returns:
        str: The formatted text with line breaks to fit within the specified width.
    """
    if isinstance(text, bytes):
      text = text.decode("utf-8")

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if current_line:
            """
            # Check if adding the next word exceeds the width limit
            current_line_length = ImageDraw.textbbox((0, 0), " ".join(current_line + [word]), font=font)[2]

            #current_line_length = font.getsize(" ".join(current_line + [word]))[0]
            if current_line_length < w * (1 - min_word_on_line):
                # Add the word to the current line
                current_line.append(word)
            else:
            """
            # Start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            # First word on the line
            current_line.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    #print(lines)
    return "\n".join(lines)

def typeset_blurb(img, blurb, transparency):
    if isinstance(blurb, TranslatedBlurb):
        text = (blurb.translation.decode("utf-8", 'ignore'))
    else:
        if isinstance(text, bytes):
            text = str(text.decode("utf-8", 'ignore'))

    if len(text) > 4:
        area = blurb.w * blurb.h
        fontsize = int(math.sqrt(area) / 10)

        if fontsize < 12:
            fontsize = 12

        usingFont = ImageFont.truetype("Arial.ttf", fontsize)

        # Split the text into lines with word wrap
        lines = textwrap.wrap(text, width=20)  # Adjust the width as needed

        if lines:
            d = ImageDraw.Draw(img)

            # Calculate the height of each line
            line_height = fontsize + 2  # Adjust as needed

            # Iterate through lines and draw a white box and text for each line
            for i, line in enumerate(lines):
                # Calculate the position of the white box for this line
                box_y = blurb.y + i * line_height
                box_coords = [(blurb.x, box_y), (blurb.x + blurb.w, box_y + line_height)]

                # Draw the white box
                filling = (255, 255, 255, transparency)
                ImageDraw.Draw(img, "RGBA").rounded_rectangle(box_coords, radius=25, fill=filling, width=0)

                # Draw the text on top of the white box
                text_coords = (blurb.x, box_y)
                d.text(text_coords, line, fill=(0, 0, 0), font=usingFont, encoding='utf-8')

def get_params():
    params = ""
    params += "--psm 5"

    configParams = []
    def configParam(param, val):
      return "-c " + param + "=" + val
    configParams.append(("chop_enable", "T"))
    configParams.append(('use_new_state_cost','F'))
    configParams.append(('segment_segcost_rating','F'))
    configParams.append(('enable_new_segsearch','0'))
    configParams.append(('textord_force_make_prop_words','F'))
    configParams.append(('tessedit_char_blacklist', '|ㆍ@ー()\、-ㅡ《0123456789}><~^/#'))
    configParams.append(('textord_debug_tabfind','0'))
    #configParams.append(('preserve_interword_spaces','1'))

    params += " ".join([configParam(p[0], p[1]) for p in configParams])
    return params

def get_blurbs(img, input_language):

  # Convert the input image to grayscale
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply adaptive thresholding to create a binary image (black and white) using a Gaussian method.
  # The thresholded image is then inverted using bitwise_not.
  img_gray = cv2.bitwise_not(cv2.adaptiveThreshold(img_gray, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 75, 10))

  # Create a 2x2 kernel to be used for erosion operation
  kernel = np.ones((2,2), np.uint8)

  # Erode the image to remove noise and small details by applying the kernel repeatedly.
  # This operation helps to make the text or blurb regions more distinguishable.
  img_gray = cv2.erode(img_gray, kernel, iterations=2)

  # Invert the image again to revert it back to the original orientation.
  img_gray = cv2.bitwise_not(img_gray)

  # Find contours in the processed image using RETR_TREE mode (hierarchical contour retrieval mode)
  # and CHAIN_APPROX_SIMPLE method (compresses horizontal, vertical, and diagonal segments and leaves only their end points).
  contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Create an empty mask image with the same dimensions as the input image.
  mask = np.zeros_like(img)

  # Convert the mask to grayscale (single-channel) for further processing.
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

  # Get the height, width, and number of channels of the input image.
  height, width, channel = img.shape

  pruned_contours = []

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100 and area < ((height / 3) * (width / 3)):
      pruned_contours.append(cnt)

  # find contours for the mask for a second pass after pruning the large and small contours
  cv2.drawContours(mask, pruned_contours, -1, (255,255,255), 1)
  contours2, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

  final_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)

  blurbs = []
  for cnt in contours2:
    area = cv2.contourArea(cnt)
    if area > 1000 and area < ((height / 3) * (width / 3)):
      draw_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
      approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
      #pickle.dump(approx, open("approx.pkl", mode="w"))
      cv2.fillPoly(draw_mask, [approx], (255,0,0))
      cv2.fillPoly(final_mask, [approx], (255,0,0))
      image = cv2.bitwise_and(draw_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
      draw_mask_inverted = cv2.bitwise_not(draw_mask)
      image = cv2.bitwise_or(image, draw_mask_inverted)
      y = approx[:, 0, 1].min()
      h = approx[:, 0, 1].max() - y
      x = approx[:, 0, 0].min()
      w = approx[:, 0, 0].max() - x
      image = image[y:y+h, x:x+w]

      padding = 10  # Number of pixels for padding

      # Add padding to the image
      image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=255)

      pil_image = Image.fromarray(image)

      #OCR recognition
      text = pytesseract.image_to_string(pil_image, lang=input_language, config=get_params())
      text = text.replace("\n", "")
      text = text.replace("\x0c", "")
      print(text)
      #Non UTF8 cleaner
      """
      if text and text.strip():
        for letter in text:
          if ord(letter) > 255:
              #print("deleting atrocity")
              text = text.replace(letter, "")"""
      if text and text.strip() and text != None:
        #filter out noise under x characters
        if len(text)>3:
          blurb = Blurb(x, y, w, h, text)
          #print(blurb)
          blurbs.append(blurb)
          #print ("Attempt: " + text + ' -> ' + str(translator.translate(text,dest='fr').text))

  return blurbs

def load_images_from_folder(folder):
    images = []
    resize_optimisation = True
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            if resize_optimisation:
              # Resize the image to a smaller resolution
              # You can adjust the target width and height as needed
              img = cv2.resize(img,(800,1000))

            images.append(img)

    return images

def load_names_from_folder(folder):
    names = []

    for filename in os.listdir(folder):
        if filename is not None:
            names.append(filename)
            print(filename)

    return names

if __name__ == "__main__":
  # Initialize parser
  parser = argparse.ArgumentParser(description="Your script description here")

  # Add required argument for input language
  parser.add_argument("-i", "--input_language", help="Input language ('jpn_vert' for Japanese vertical) more at: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html", choices=['jpn_vert'], required=True)

  # Add optional argument for output language with a default of 'fr'
  parser.add_argument("-o", "--output_language", help="Output language ('fr' by default)", choices=['fr'], default='fr')

  # Add optional arguments with default values
  parser.add_argument("-if", "--input_folder", help="Input folder path", default="./original")
  parser.add_argument("-of", "--output_folder", help="Output folder path", default="./translated")
  parser.add_argument("-t", "--transparency", help="Transparency (between 1 and 255)", type=int, default=128)

  # Parse the command line arguments
  args = parser.parse_args()

  # Access the values of the arguments
  input_folder = args.input_folder
  output_folder = args.output_folder
  transparency = args.transparency
  input_language = args.input_language
  output_language = args.output_language

  # Now you can use these values in your script
  print(f"Input folder path: {input_folder}")
  print(f"Output folder path: {output_folder}")
  print(f"Transparency: {transparency}")
  print(f"Input language: {input_language}")
  print(f"Output language: {output_language}")

  images  = load_images_from_folder(input_folder)
  names = load_names_from_folder(input_folder)
  transImg = []
  i = 0
  b = 0

  for img in images:
    blurbs = get_blurbs(img, input_language)
    needTransImg = Image.fromarray(img.copy())

    for blurb in blurbs:
      b = b + 1
      #try:
      translated = translate_blurb(blurb, output_language)
      typeset_blurb(needTransImg, translated, transparency)
      trans = needTransImg

    save_file = os.path.join(output_folder,str(i) + '.jpg')
    print(save_file)
    trans.save(save_file, format='JPEG')
    i = i+1

  print('Number of translated blurb:' + str(b))
  print('Number of translated pages:' + str(i))