import dbconnect
import json

def getgeboekteKamerinfo(abc):
    mycursor = dbconnect.con.cursor()

    mycursor.execute("SELECT * FROM boeking")

    myresult = mycursor.fetchall()

    ab=json.dumps(myresult, indent=4, sort_keys=True, default=str)
    #ab=json.dumps( [dict(ix) for ix in myresult] )
    print(ab)
    return ab

