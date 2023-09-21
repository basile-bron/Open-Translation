# Open-Translation

## Download Tesseract here :
https://github.com/UB-Mannheim/tesseract/wiki


Edit the path of tesseract in the script:
```
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

```

## PSM reminder
"""
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
"""


## available languages at :
https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
