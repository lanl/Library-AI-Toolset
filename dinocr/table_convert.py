import pandas as pd

# 'ocr_model' requires an easyocr Reader object
def table_to_markdown(image, ocr_model):   
    result = ocr_model.readtext(image)
    
    cells = [([r[0][0][0], r[0][0][1], r[0][2][0], r[0][2][1]], r[1]) for r in result]
    bboxes = [c[0] for c in cells]

    min_x = min([bbox[0] for bbox in bboxes])
    min_y = min([bbox[1] for bbox in bboxes])
    max_x = max([bbox[2] for bbox in bboxes])
    max_y = max([bbox[3] for bbox in bboxes])

    x_buckets = {}

    # How do we algorithmically determine this?
    x_thresh = (max_x - min_x) / 10
    
    for c in cells:
        center_x = ((c[0][2]+c[0][0])/2)
        bbox = c[0]
        data = c[1]
        if not x_buckets:
            x_buckets[center_x] = [(bbox, data)]
        else:
            flag = False
            for key in x_buckets.keys():
                if ((center_x - x_thresh) <= key) and ((center_x + x_thresh) >= key):
                    if (data, bbox) not in x_buckets[key]:
                        x_buckets[key].append((bbox, data))
                    flag = True
                    break
            if not flag: x_buckets[center_x] = [(bbox, data)]
    
    x_buckets = dict(sorted(x_buckets.items()))

    y_buckets = {}

    # How do we algorithmically determine this?
    y_thresh = (max_y - min_y) / 100
    
    for c in cells:
        center_y = ((c[0][1]+c[0][3])/2)
        bbox = c[0]
        data = c[1]
        if not y_buckets:
            y_buckets[center_y] = [(bbox, data)]
        else:
            flag = False
            for key in y_buckets.keys():
                if ((center_y - y_thresh) <= key) and ((center_y + y_thresh) >= key):
                    if (data, bbox) not in y_buckets[key]:
                        y_buckets[key].append((bbox, data))
                    flag = True
                    break
            if not flag: y_buckets[center_y] = [(bbox, data)]
    
    y_buckets = dict(sorted(y_buckets.items()))

    dataframe_list = [[None for _ in range(len(x_buckets))] for _ in range(len(y_buckets))]

    for c in cells:
    
        x_coord = None
        y_coord = None
        
        for ind_x, bucket in enumerate(x_buckets):
            if c in x_buckets[bucket]:
                x_coord = ind_x
                break
    
        for ind_y, bucket in enumerate(y_buckets):
            if c in y_buckets[bucket]:
                y_coord = ind_y
                break

        dataframe_list[y_coord][x_coord] = c[1]
    
    df = pd.DataFrame(dataframe_list)
    return df.to_markdown(index=False)
