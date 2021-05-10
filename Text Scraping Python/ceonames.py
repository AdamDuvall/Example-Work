
# first save your RTF file as plan txt file
# then you can open it like this
filedata = []                            
with open (r"\Users\Adam\Desktop\Scraping\NY38013900TXT.txt", "r") as file:
    for line in file:                
        filedata.append(line)          
print(filedata)                           

# print the CEO name of the first company
print(filedata[21])

# define an empty variable
ceoname = []

# then loop over each line in your file
for linenumber in range(1, len(filedata), 1):
    if filedata[linenumber].find("Chief Executive:") == 0:  # if the words 'Chief Executive:' is included in the line
        ceoname.append(filedata[linenumber])                # append it to the ceoname variable
    else:                                                   # otherwise continue to loop through the text
        continue

# this is the resulting list
ceoname

from pandas import DataFrame
ceoname2=DataFrame(ceoname)
ceoname2.to_excel(r'/Users/Adam/Desktop/NY38013900EXEC.xlsx', sheet_name='NY38013900', index = False)
