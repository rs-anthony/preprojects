import dbconnect
import json

def checkEmailadress(kamerid):
    mycursor = dbconnect.con.cursor()

    sql = "SELECT emailadress FROM hotel_database.member"
    val = (kamerid,)

    mycursor.execute(sql)

    myresult = mycursor.fetchall()

    ab=json.dumps(myresult)
    #ab=json.dumps( [dict(ix) for ix in myresult] )
    return ab