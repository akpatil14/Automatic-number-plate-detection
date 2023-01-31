import csv
 
def db_get_val(b):
  
    with open('db.csv') as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
            
            if(row['Vehicle']==b):
                return (row['Name'],row['Contact'],row['Address'],row['Status']) 

        return ('0','0','0','0')

Name, Contact, Address,Status=db_get_val('29A33185')
print(Name, Contact,Address,Status)


      


#update_vote('BJP',2)    #2 is integer val

#a=b'40003539FBB7'
#[cost,Name]=db_get_val(a.decode('ascii'))
#print (cost)
#print (Name)

    
