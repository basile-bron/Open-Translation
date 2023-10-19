# Open-Translation

![banner](banner_github.jpg)

## About

Manga page translation. get a translated image or a json conataining the dialogue output with it size and location.

## installation

### Download Tesseract here :
https://github.com/UB-Mannheim/tesseract/wiki


### Edit the path of tesseract in the script:
```
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

```
pipenv install
```

## Command line

### Option and usage
```
usage: trad.py [-h] [-i INPUT_LANGUAGE] [-o OUTPUT_LANGUAGE] [-if INPUT_FOLDER]
               [-of OUTPUT_FOLDER] [-t TRANSPARENCY] [-j JSON]

options:
  -h, --help            show this help message and exit
  -i INPUT_LANGUAGE, --input_language INPUT_LANGUAGE
                        Input language ('jpn_vert' for Japanese vertical) more     
                        at: https://tesseract-ocr.github.io/tessdoc/Data-Files-    
                        in-different-versions.html
  -o OUTPUT_LANGUAGE, --output_language OUTPUT_LANGUAGE
                        Output language ('en' by default)
  -if INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Input folder path
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Output folder path
  -t TRANSPARENCY, --transparency TRANSPARENCY
                        Transparency (between 1 and 255)
  -j JSON, --json JSON  return only the translated text
```

### example
```
& C:/Python310/python.exe ./trad.py  -i 'jpn_vert' -if "./original/001.jpg"
```

## Additionnal informations

### Json output :
  instead of getting a translated image you can ask for a json file containing for each image i.g for output language = "en" you will get for each text buble:

  ```
  "x"
  "y"
  "w"
  "h"
  "original_text"
  "text_fr"
  "font_size_fr
  ```



  Cords and sizes are a ratio not a pixel size.
  i.e  height 0.5 is a buble of half of the height of the original image.
  it allows you to render the translation on any size and scale of the original image.

## PSM reminder
```
    Page segmentation modes:
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
    10    Treat the image as a single character.
    11    Sparse text. Find as much text as possible in no particular order.
    12    Sparse text with OSD.
    13    Raw line. Treat the image as a single text line,
```

## available languages at :
https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
