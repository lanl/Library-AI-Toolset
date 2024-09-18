import requests, io, os
import pymupdf
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image, ImageDraw
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import pytesseract
import easyocr
from pathlib import Path
from table_convert import table_to_markdown
import numpy as np
pytesseract.pytesseract.tesseract_cmd = <pytesseract path>

# placeholder: If we don't have English documents, we can easyocr their tables here
def load_lang_easyocr():
    pass

def load_all_models():
    reader = easyocr.Reader(['en'])
    tex_model = load_model()
    tex_processor = load_processor()
    doc_processor = AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet")
    doc_model = DeformableDetrForObjectDetection.from_pretrained("Aryn/deformable-detr-DocLayNet")
    
    return reader, tex_model, tex_processor, doc_processor, doc_model
    
def extract(pdf, output, verbose=False):
    
    print("Loading models...")
    reader, tex_model, tex_processor, doc_processor, doc_model = load_all_models()
    
    print("Models loading. Extracting content...")
    Path(output).mkdir(parents=True, exist_ok=True)
    if os.path.exists(output+"output.mkd"):
        os.remove(output+"output.mkd")
    f = open(output+"output.mkd", "w")
    
    doc = pymupdf.open(pdf)
    
    for page_count, page in enumerate(doc):
        print("Processing Page",page_count,"/",len(doc))
        pixmap = page.get_pixmap(dpi=300)
        img = pixmap.tobytes()
        image = Image.open(io.BytesIO(img))
                
        ### Step 1: Read page data into the transformer model
        inputs = doc_processor(images=image, return_tensors="pt")
        outputs = doc_model(**inputs)
        
        ### Step 2: Break down page into elements
        confidence_thresh = 0.60
        column_thresh = 600

        column_dict = {}
        
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections above our confidence threshold
        target_sizes = torch.tensor([image.size[::-1]])
        results = doc_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_thresh)[0]
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

            label = doc_model.config.id2label[label.item()]
            box = [round(i, 2) for i in box.tolist()]
            min_x = box[0]
            if not column_dict: column_dict[min_x] = [(label, box)] # priming the dict
            
            flag = False
            for key in column_dict.keys():
                if ((min_x - column_thresh) <= key) and ((min_x + column_thresh) >= key):
                    if (label,box) not in column_dict[key]:
                        column_dict[key].append((label, box))
                    flag = True
                    break
            if not flag: column_dict[min_x] = [(label, box)]
            
            if verbose:
                print(
                    f"Detected {label} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
        
        # Sort by y-axis, allowing printing of text in-order
        for key in column_dict:
            column_dict[key] = sorted(column_dict[key], key=lambda box: (box[1][1]))
            
        ### Step 3: Convert elements to markdown/latex format
        img_count = 0
        table_count = 0

        caption_queue = None
        picture_queue = None
        
        for key in sorted(column_dict.keys()):
            for tup in column_dict[key]:
                label = tup[0]
                #print(label, tup[1])
                if label in ["Title", "Text", "Formula", "List-item"]:
                    results = batch_inference([image.crop(tup[1])], tex_model, tex_processor)
                    f.write(results[0])
                    if label == "Title":
                        f.write('\n') # workaround for title formatting
                elif label == "Section-header":
                    f.write("### "+ reader.readtext(np.array(image.crop(tup[1])), detail = 0, paragraph=True)[0])
                    f.write('\n')
                elif label == "Picture":
                    filename = "image_{}_{}".format(page_count, img_count)
                    image.crop(tup[1]).save(output+filename+".png")
                    img_count += 1
                    f.write("![{}]({}.png)".format(filename, filename))
                    f.write('\n')
                    
                    ### This caption extraction and tying it to images can be tricky.
                    ### It includes situations where captions appear before and after
                    ### an image. We'll have to consider that in the future.
                    
                    #f.write("![{}]({}.png \'{}\')".format(filename, filename))
                    # filename = "image_{}_{}".format(page, count)
                    # image.crop(tup[1]).save(output_dir+filename+".png")
                    # count += 1
                    # if caption_queue is None:
                    #     if picture_queue is not None:
                    #         # In this case, we have an earlier image with no caption; go ahead and print it to the markdown file
                    #         f.write("![{}]({}.png \'{}\')".format(filename, filename, caption_queue))
                    #     picture_queue = filename
                    # else:
                    #     # We're only here if there's a caption and no image in the picture_queue
                    #     f.write("![{}]({}.png \'{}\')".format(filename, filename, caption_queue))
                    #     caption_queue = None
                    
                elif label == "Caption":
                    caption = reader.readtext(np.array(image.crop(tup[1])), detail = 0, paragraph=True)[0].strip()
                    f.write(caption)
                    
                    # if picture_queue is None:
                    #     # We assume there are no rogue captions in these docs; might change this if that's not the case
                    #     caption_queue = caption
                    # else:
                    #     f.write("![{}]({}.png \'{}\')".format(filename, filename, caption))
                    #     picture_queue = None
                    
                elif label == "Table":
                    filename = "table_{}_{}".format(page_count, table_count)
                    image.crop(tup[1]).save(output+filename+".png")
                    f.write(table_to_markdown(output+filename+".png", reader))
                    f.write('\n')
                    table_count += 1

    f.close()

if __name__ == "__main__":    
    extract('examples/Example_Doc.pdf', output='output/')
