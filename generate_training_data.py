# In[281]:

import xlrd
from preprocess_data_putusan import *

book = xlrd.open_workbook('C:/Users/Alvin/Downloads/t_d_pasal_amar_putusan (1).xlsx')

data_sheet=book.sheet_by_index(0)

sample_data=data_sheet.row_values(0)[0].encode('ascii','ignore')

cells = data_sheet.col_slice(start_rowx=0,colx=0)

myfile = open('data_putusan_raw.txt', 'w')

for cell in cells:
    data = cell.value.encode('ascii','ignore')
    data = preprocess_string(data)
    for word in data:
    	 myfile.write("%s\n" % word)
    myfile.write("\n")

myfile.close()