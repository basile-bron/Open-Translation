import sys
import cv2
import os
import re
import math
import argparse
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
import textwrap
import logging
import json
import cProfile


#install new llanguage documentation:
#https://ocrmypdf.readthedocs.io/en/latest/languages.html
#example :  
#cd C:\Program Files\Tesseract-OCR\tessdata
#curl -o chi_sim.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata
 

googletrans = True
if googletrans:
  from googletrans import Translator
  translator = Translator()
else:
  #to coment if googletrans
  from translate import Translator
  translator= Translator(to_lang="english")

class Blurb(object):
    def __init__(self, x, y, w, h, text):
      """
      Initialize a Blurb object.

      Args:
          x (int): X-coordinate of the top-left corner of the region.
          y (int): Y-coordinate of the top-left corner of the region.
          w (int): Width of the region.
          h (int): Height of the region.
          text (str): The actual text content within the region.
      """
      self.x = x
      self.y = y
      self.w = w
      self.h = h
      self.text = text

    def clean_text(self):
      text = self.text
      #Non UTF8 cleaner
      #if text and text.strip():
      #  for letter in text:
      #    if ord(letter) > 255:
      #        text = text.replace(letter, "")
      text = re.sub(r"\n", "", text)

      # Define a regular expression pattern to allow only specific characters
      #allowed_characters_pattern = re.compile(r'[a-zA-Z0-9., !?]+')
      
      # Keep only allowed characters by applying the regular expression pattern
      #text = ''.join(allowed_characters_pattern.findall(text))

      return text

    def __str__(self):
      """
      Return a string representation of the Blurb object.

      Returns:
          str: The string representation in the format: "x,y w x h : text".
      """
      return str(self.x) + ',' + str(self.y) + ' ' + str(self.w) + 'x' + str(self.h) + ' ' + self.text

def clean_trans_output(text):

  #if text and text.strip():
  #  for letter in text:
  #    if ord(letter) > 255:
  #        text = text.replace(letter, "")
  # Remove special characters and symbols
  text = re.sub(r'[^a-zA-Z0-9\s\'-]', '', text)

  # Remove extra spaces and leading/trailing spaces
  text = ' '.join(text.split())


  return text

class TranslatedBlurb(Blurb):
    def __init__(self, x, y, w, h, text, translation):
      """
      Initialize a TranslatedBlurb object.

      Args:
          x (int): X-coordinate of the top-left corner of the region.
          y (int): Y-coordinate of the top-left corner of the region.
          w (int): Width of the region.
          h (int): Height of the region.
          text (str): The actual text content within the region.
          translation (str): The translated text.
      """
      Blurb.__init__(self, x, y, w, h, text)
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
      return cls(parent.x, parent.y, parent.w, parent.h, parent.text, translation)

def translate_blurb(blurb, output_language):
    """
    Translate the text content of a Blurb object to English and create a TranslatedBlurb object.

      Args:
          blurb (Blurb): The Blurb object to be translated.

      Returns:
        blurb (Blurb): The Blurb object translated.
    """
    
    try:
      if googletrans:
        translated_text = translator.translate(blurb.clean_text(), dest=output_language)
        translation = translated_text.text.encode('utf-8', 'ignore')
      else:
        translated_text = translator.translate(blurb.clean_text())
        translation = translated_text.encode('utf-8', 'ignore')

    except Exception as e:
       logging.error(e)
       logging.error(blurb.clean_text())
       translation = ''

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

    return "\n".join(lines)

def typeset_blurb(img, blurb, transparency :int):
    if isinstance(blurb, TranslatedBlurb):
        text = (blurb.translation.decode("utf-8", 'ignore'))
    else:
        if isinstance(text, bytes):
            text = str(text.decode("utf-8", 'ignore'))

    # Check if there are exactly three words
    if len(text.split()) > 4:
        area = blurb.w * blurb.h
        fontsize = int(math.sqrt(area) / 10)

        if fontsize < 12:
            fontsize = 12

        usingFont = ImageFont.truetype("C:/Users/Skull/Documents/github/Open-Translation/Arial.ttf", fontsize)

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

def get_params(mode):
    params = ""
    params += "--psm " + str(mode)

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

def get_blurbs(img, input_language, ocr_mode):

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
      text = pytesseract.image_to_string(pil_image, lang=input_language, config=get_params(ocr_mode))
      text = text.replace("\n", "")
      text = text.replace("\x0c", "")      
      if text and text.strip() and text != None:
        #filter out noise under x characters
        if len(text)>3:
          blurb = Blurb(x, y, w, h, text)
          #print(blurb)
          blurbs.append(blurb)
          #print ("Attempt: " + text + ' -> ' + str(translator.translate(text,dest='fr').text))

  # Get the dimensions (size) of the input image
  height, width = img.shape[:2]

  return blurbs, height, width

def load_images_from_folder(folder):
    images = []
    names = []

    resize_optimisation = False
    #for filename in os.listdir(folder):
    # List of valid image file extensions
    valid_image_extensions = ['.jpg', '.jpeg', '.png','.JPG','.JPEG',]

    # Iterate through files in the folder
    for filename in os.listdir(folder):
        # Get the file extension of the current file
        file_extension = os.path.splitext(filename)[1].lower()

        # Check if the file extension is in the list of valid image extensions
        if file_extension in valid_image_extensions:
          names.append(filename)
          img = cv2.imread(os.path.join(folder,filename))
          if img is not None:
              if resize_optimisation:
                # Resize the image to a smaller resolution
                # You can adjust the target width and height as needed
                img = cv2.resize(img,(243,358))

              images.append(img)
    
    logging.info("filenames: %s", names)

    return images, names

# Configure logging as mentioned in the previous answer
def log_exception(exc_type, exc_value, exc_traceback):
  # Log the exception with the custom log level "ANY"
  logging.log(100, "An exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

def main():

  logging.debug('Script starting')

  # Initialize parser
  parser = argparse.ArgumentParser(description="Your script description here")

  # Add required argument for input language
  parser.add_argument("-i", "--input_language", help="Input language ('jpn_vert' for Japanese vertical) more at: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html", default='jpn_vert')
  parser.add_argument("-o", "--output_language", help="Output language ('en' by default)",type=str, default='en')
  parser.add_argument("-if", "--input_folder", help="Input folder path", default="./original")
  parser.add_argument("-of", "--output_folder", help="Output folder path", default="./translated")
  parser.add_argument("-t", "--transparency", help="Transparency (between 1 and 255)", type=int, default=200)
  parser.add_argument("-j", "--json", help="return only the translated text", type=bool, default=True)

  try:
    args = parser.parse_args()
    logging.debug(args)
  except argparse.ArgumentError as e:
    # Log the specific argument that caused the error
    logging.error(f"Invalid argument: {e.argument_name}")
    # Log the received arguments
    logging.debug(f"Received arguments: {sys.argv[1:]}")
  except Exception as e:
    logging.error(f"An error occurred while parsing arguments: {e}")

    
  # Access the values of the arguments
  input_folder = args.input_folder
  output_folder = args.output_folder
  transparency = args.transparency
  input_language = args.input_language
  output_language = args.output_language
  output_json = args.json

  # Define a mapping of language aliases to Tesseract language codes
  language_mapping = {
      'chi_sim': 'chi_sim_vert',
      'chinese': 'chi_sim_vert',
      'china': 'chi_sim_vert',
      'japanese': 'jpn_vert',
      'japan': 'jpn_vert',
      # Add more aliases and language codes as needed
  }

  # Get the user input for the language parameter
  user_input = input_language.strip().lower()

  # Use the mapping to determine the Tesseract language code
  if user_input in language_mapping:
      input_language = language_mapping[user_input]
  else:
      # Default to 'eng' (English) if the input doesn't match any alias
      input_language = 'jpn_vert'
  
  ocr_mode = 5
  if 'vert' not in input_language:
    ocr_mode = 6
     
  # Now you can use these values in your script
  logging.info("arguments : -if: "+ str(input_folder) + " -of: "+ str(output_folder) + " -t: "+ str(transparency) + " i: " + str(input_language) + " o: "+ str(output_language) + " json: "+ str(output_json))

  # Check if input_folder is a folder or a single image file
  if os.path.isdir(input_folder):
      logging.info("input path is a folder")
      # Load images from the folder
      images, names = load_images_from_folder(input_folder)
  elif os.path.isfile(input_folder):
      logging.info("input path is a file")
      # Input is a single image file
      img = cv2.imread(os.path.join(input_folder))
      if img is not None:
        images = [cv2.resize(img,(800,1000))]
      names = [os.path.basename(input_folder)]
  else:
      logging.error("Input folder or file not found.")
      exit(1)

  # Check if the folder exists
  if not os.path.exists(output_folder):
      # If it doesn't exist, create it
      os.makedirs(output_folder)

  # Check if the output JSON file already exists
  output_json_file = os.path.join(output_folder, "translations.json")
  translations = {}

  if os.path.exists(output_json_file):
      # If the JSON file exists, load its contents into the translations dictionary
      with open(output_json_file, "r", encoding="utf-8") as json_file:
        try:
          translations = json.load(json_file)
        except Exception as e:
            # Handle JSON loading errors here
            logging.error(f"Error loading JSON file: {str(e)}")
  i = 0
  b = 0

  for img in images:
    image_filename = names[i]
    
    # don't translate if already translated in the requested language
    if 'text_'+ str(output_language) not in translations.get(image_filename, {}):
      
      # ocr already output original text previously, just do a simple translation 
      if 'original_text_debug' in translations.get(image_filename, {}):
        logging.debug("simple translation")
        # TO DO
      else:
        logging.debug("full translation")

        blurbs, height, width = get_blurbs(img, input_language, ocr_mode)

        for blurb in blurbs:
          logging.debug("blurb number : %s", b)
          logging.debug("out of : %s", len(blurbs))

          b += 1
          if output_json:
            # Translate the blurb and store the translation information in the dictionary
            translated = translate_blurb(blurb, output_language)
            text = clean_trans_output(str(translated.translation.decode("utf-8", 'ignore')))
            if translated.translation != '' and len(text.split()) > 3:
              #if meta already collected
              #if str(blurb.text) in translations.get(image_filename, {}):
              
              translation_info = {
                  "x": round(float(translated.x / width), 2),
                  "y": round(float(translated.y / height), 2),
                  "w": round(float(translated.w / width), 2),
                  "h": round(float(translated.h / height), 2),
                  "original_text": str(blurb.text),
                  "text_"+ str(output_language): text,
                  "font_size_"+ str(output_language): round(math.sqrt( (translated.w * translated.h) / len(text) ))
              }

              # Use the image filename as a unique key for each image in the translations dictionary
              if image_filename not in translations:
                try:
                  translations[image_filename] = []
                except:
                  translations[image_filename] = translation_info
                  
              translations[image_filename].append(translation_info)
          else:
            needTransImg = Image.fromarray(img.copy())
            typeset_blurb(needTransImg, translated, transparency)
            trans = needTransImg

        if output_json:
          # Save the updated translations to the JSON file
          with open(output_json_file, "w", encoding="utf-8") as json_file:
            logging.debug(translations)
            try:
              # Attempt to serialize the translation_info to JSON
              json.dump(translations, json_file, ensure_ascii=False, indent=4)
            except Exception as e:
              # Handle the exception, print debugging information, or remove the problematic data
              logging.error(str(e))
        else:
          # Save the translated image with the image filename as the name
          save_file = os.path.join(output_folder, str(i) + '.jpg')
          logging.info(save_file)
          trans.save(save_file, format='JPEG')
    else:
      logging.info('Already translated: '+ image_filename)
    i += 1

  print('Number of translated blurb:' + str(b))
  print('Number of translated pages:' + str(i))

  logging.debug('Script end')

if __name__ == "__main__":

  logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename='trad.log',  # Specify a log file
    format='%(asctime)s [%(levelname)s]: %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date format
  )

  # Set the custom exception handler
  sys.excepthook = log_exception
  
  # edit the path of tesseract to yours:
  pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

  main()
  # to uncomment for benchmark and optimisation review
  #cProfile.run('main()')
