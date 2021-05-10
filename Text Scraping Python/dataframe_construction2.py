
#%%

def mergent_to_df(textfilelist):
    #file_list_ny = ["nydata1.txt", "nydata2.txt", "nydata3.txt", "nydata4.txt", "nydata5.txt", "nydata6.txt",]
    #file_list_pa = ["patext.txt"]
    master_df = pd.DataFrame()
    #Corporate family
    for data in textfilelist:
        df = pd.DataFrame()
        filedata = []
        with open ("C:/Users/ctyle//OneDrive/Desktop/" + data, "r") as file:
            for line in file:
                filedata.append(line)
                
        companies = [[]]
        
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Company Overview") == 0: 
                print(filedata[linenumber + 2])
                companies[-1].append(filedata[linenumber + 2]) #Append The name of the coporation to the list    
            elif filedata[linenumber].find("Business Desc:") == 0:
                companies[-1].append(filedata[linenumber - 4].strip("\n")) # This is the company in the coporate family
            elif filedata[linenumber].find("Sales Volume:") == 0:
                companies.append([])                  # Once all of the companies in the corporate family have been added, append another list
            else:       
                continue
            
            
        prime_corp = []
        
        for i in companies:
            if len(i) > 0:
                prime_corp.append(i[0])
                i.pop(0)
            
        print(prime_corp, "Prime corporations")
        
        companies.pop(-1)
        
        df['Company'] = prime_corp
        df['Corporate Family'] = companies
        
        ceoname = []
        
        # then loop over each line in your file
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Chief Executive:") == 0:  # if the words 'Chief Executive:' is included in the line
                ceoname.append(filedata[linenumber][17:]) # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
            
        df['Executive'] = ceoname
        
        #Year Founded:
            
        year = []
        
        # then loop over each line in your file
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Year Founded:") == 0:  # if the words 'Chief Executive:' is included in the line
                year.append(filedata[linenumber][13:])                # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
        
        print(year)
        if len(year) != len(df):
            year = []
            for linenumber in range(1, len(filedata), 1):
                if filedata[linenumber].find("Chief Executive:") == 0:
                    if filedata[linenumber - 4].find("Year Founded:") == 0:# if the words 'Chief Executive:' is included in the line
                        year.append(filedata[linenumber - 4][13:])                # append it to the ceoname variable
                    else:                                                   # otherwise continue to loop through the text
                        year.append("NA")
            
        
        df["Year Founded"] = year
        
        #Employees
        
        employees = []
        
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Employs:") == 0:  # if the words 'Chief Executive:' is included in the line
                employees.append(filedata[linenumber][9:])                # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
            
        df["Employees"] = employees
        
        #SIC code
        
        # Industry Codes
        # Primary SIC Code
        # 8082 - Home health care services
        # Secondary SIC Codes
        # 8082 - Home health care services
        # NAICS Code
        # 621610 - Home Health Care Services
        
        SIC = []
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Primary SIC Code") == 0:  # if the words 'Chief Executive:' is included in the line
                SIC.append(filedata[linenumber + 1][0:5])                # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
            
        df["Primary SIC code"] = SIC
        
        #Sales volume
        
        #Sales Volume: $4,563,943
        sales_volume = []
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Sales Volume:") == 0:  # if the words 'Chief Executive:' is included in the line
                sales_volume.append(filedata[linenumber][15:])                # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
            
        df["Sales Volume"] = sales_volume
        
        #Phone Number
        
        #Telephone: 518 489-2681
        
        phone = []
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Telephone:") == 0:  # if the words 'Chief Executive:' is included in the line
                phone.append(filedata[linenumber][11:])                # append it to the ceoname variable
            else:                                                   # otherwise continue to loop through the text
                continue
            
        df["Phone Number"] = phone
        
        #Financial Condition:
            
        #Financial Condition: UNBALANCED
        #This only returns information on a small subset of companies
        fin_cond = []
        for linenumber in range(1, len(filedata), 1):
            if filedata[linenumber].find("Chief Executive:") == 0:
                if filedata[linenumber + 14].find("Financial Condition:") == 0:# if the words 'Chief Executive:' is included in the line
                    fin_cond.append(filedata[linenumber + 14][21:])                # append it to the ceoname variable
                else:                                                   # otherwise continue to loop through the text
                    fin_cond.append("NA")
        df["Financial Condition"] = fin_cond
        
        print("This is the end of file", data)
        master_df = master_df.append(df)
        return(master_df)
    

file_list_ny = ["nydata1.txt", "nydata2.txt", "nydata3.txt", "nydata4.txt", "nydata5.txt", "nydata6.txt",]
master_df = mergent_to_df(file_list_ny)
#%%
# import pandas as pd

# file_list_ny = ["nydata1.txt", "nydata2.txt", "nydata3.txt", "nydata4.txt", "nydata5.txt", "nydata6.txt",]
# #file_list_pa = ["patext.txt"]
# master_df = pd.DataFrame()
# #Corporate family
# for data in file_list_ny:
#     df = pd.DataFrame()
#     filedata = []
#     with open ("C:/Users/ctyle//OneDrive/Desktop/" + data, "r") as file:
#         for line in file:
#             filedata.append(line)
            
#     companies = [[]]
    
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Company Overview") == 0: 
#             print(filedata[linenumber + 2])
#             companies[-1].append(filedata[linenumber + 2]) #Append The name of the coporation to the list    
#         elif filedata[linenumber].find("Business Desc:") == 0:
#             companies[-1].append(filedata[linenumber - 4]) # This is the company in the coporate family
#         elif filedata[linenumber].find("Sales Volume:") == 0:
#             companies.append([])                  # Once all of the companies in the corporate family have been added, append another list
#         else:       
#             continue
        
        
#     prime_corp = []
    
#     for i in companies:
#         if len(i) > 0:
#             prime_corp.append(i[0])
#             i.pop(0)
        
#     print(prime_corp, "Prime corporations")
    
#     companies.pop(-1)
    
#     df['Company'] = prime_corp
#     df['Corporate Family'] = companies
    
#     ceoname = []
    
#     # then loop over each line in your file
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Chief Executive:") == 0:  # if the words 'Chief Executive:' is included in the line
#             ceoname.append(filedata[linenumber][17:]) # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
        
#     df['Executive'] = ceoname
    
#     #Year Founded:
        
#     year = []
    
#     # then loop over each line in your file
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Year Founded:") == 0:  # if the words 'Chief Executive:' is included in the line
#             year.append(filedata[linenumber][13:])                # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
    
#     print(year)
#     if len(year) != len(df):
#         year = []
#         for linenumber in range(1, len(filedata), 1):
#             if filedata[linenumber].find("Chief Executive:") == 0:
#                 if filedata[linenumber - 4].find("Year Founded:") == 0:# if the words 'Chief Executive:' is included in the line
#                     year.append(filedata[linenumber - 4][13:])                # append it to the ceoname variable
#                 else:                                                   # otherwise continue to loop through the text
#                     year.append("NA")
        
    
#     df["Year Founded"] = year
    
#     #Employees
    
#     employees = []
    
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Employs:") == 0:  # if the words 'Chief Executive:' is included in the line
#             employees.append(filedata[linenumber][9:])                # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
        
#     df["Employees"] = employees
    
#     #SIC code
    
#     # Industry Codes
#     # Primary SIC Code
#     # 8082 - Home health care services
#     # Secondary SIC Codes
#     # 8082 - Home health care services
#     # NAICS Code
#     # 621610 - Home Health Care Services
    
#     SIC = []
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Primary SIC Code") == 0:  # if the words 'Chief Executive:' is included in the line
#             SIC.append(filedata[linenumber + 1][0:5])                # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
        
#     df["Primary SIC code"] = SIC
    
#     #Sales volume
    
#     #Sales Volume: $4,563,943
#     sales_volume = []
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Sales Volume:") == 0:  # if the words 'Chief Executive:' is included in the line
#             sales_volume.append(filedata[linenumber][15:])                # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
        
#     df["Sales Volume"] = sales_volume
    
#     #Phone Number
    
#     #Telephone: 518 489-2681
    
#     phone = []
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Telephone:") == 0:  # if the words 'Chief Executive:' is included in the line
#             phone.append(filedata[linenumber][11:])                # append it to the ceoname variable
#         else:                                                   # otherwise continue to loop through the text
#             continue
        
#     df["Phone Number"] = phone
    
#     #Financial Condition:
        
#     #Financial Condition: UNBALANCED
#     #This only returns information on a small subset of companies
#     fin_cond = []
#     for linenumber in range(1, len(filedata), 1):
#         if filedata[linenumber].find("Chief Executive:") == 0:
#             if filedata[linenumber + 14].find("Financial Condition:") == 0:# if the words 'Chief Executive:' is included in the line
#                 fin_cond.append(filedata[linenumber + 14][21:])                # append it to the ceoname variable
#             else:                                                   # otherwise continue to loop through the text
#                 fin_cond.append("NA")
#     df["Financial Condition"] = fin_cond
    
#     print("This is the end of file", data)
#     master_df = master_df.append(df)
    
# master_df.to_excel("C:/Users/ctyle//OneDrive/Desktop/padata3.xlsx")