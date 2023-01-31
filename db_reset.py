import csv      
with open('db.csv', 'w') as wfile:
    fieldnames = ['id','Wheat', 'Rice','Pulse','Edible_oil','Kerosene',]
    writer = csv.DictWriter(wfile, fieldnames=fieldnames)

    writer.writeheader()
    i=1
    while i<7:
       
        writer.writerow({'id':i, 'Wheat': '5', 'Rice': '5','Pulse': '5','Edible_oil': '5','Kerosene': '5',})
        i=i+1;

    print("database RESETED")


