import numpy
import numpy as np
import pandas as pd
import csv
                                                                                                                                                            
def removeOutliers(arr, nsigma):

    elements = numpy.array(arr)

    mean = numpy.mean(elements, axis=0)
    sd = numpy.std(elements, axis=0)

    final_list = [x for x in arr if (x > mean - nsigma * sd)]
    final_list = [x for x in final_list if (x < mean + nsigma * sd)]
    return final_list


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

def remove_by_median_outliers(data):
    source_data = data
    angle = pd.DataFrame(data,columns = ['A'])
    print (angle['A'].values)
    angle['u_medf'] = get_median_filtered(angle['A'].values, threshold=3)
    outlier_idx = np.where(angle['u_medf'].values != angle['A'].values)[0]
    print (outlier_idx)
    try:
    	for i in outlier_idx:
    		del data[i]
    	print (data)

    except Exception as e:
        print (e)
   
    row_data = [source_data,data]
    # with open('/home/signzy-engine/abs/ocr/EAST/outliers_list.csv', "a") as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row_data)
    #     fp.close
    return data


# if __name__ == "__main__":
# 	data = [87,87,85,0]

# 	d = remove_by_median_outliers(data)
# 	print d