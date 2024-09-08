import mysql.connector
import time

def sendKamerreservering(kamerid):
    con = mysql.connector.connect(
        host="",  #port erbij indien mac
        user="",
        password="",
        database="hotel_database"
    )

    mycursor = con.cursor()
    
    sql = "UPDATE hotelkamer SET reservering = 1 WHERE kamer_id = %s"
    val = (kamerid,)
    mycursor.execute(sql, val)

    reserveringmoment = int(time.time())
    sql2 = "UPDATE hotelkamer SET tijd = "+str(reserveringmoment)+" WHERE kamer_id = %s"
    mycursor.execute(sql2, val)

    #print(val,flush=True)
    
    con.commit()
    print(mycursor.rowcount, "Kamer gereserveerd")
    return "gereserveerd"

#sendKamerinfo("test1")
