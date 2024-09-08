import mysql.connector

def sendKamerboeking(kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet):
    con = mysql.connector.connect(
        host="",  #port erbij indien mac
        user="",
        password="",
        database="hotel_database"
    )

    mycursor = con.cursor()
    
    sql = "INSERT INTO boeking (kamer_id, totaalprijs, boekingsdatum_begin, boekingsdatum_eind, member_id, betaalmethode) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet)
    print(val)
    mycursor.execute(sql, val)

    con.commit()

    sql = "SELECT * FROM boeking ORDER BY boeking_id DESC"   

    mycursor.execute(sql)
    
    myresult = mycursor.fetchall()
    return [str(myresult[0][0]), str(kamerid), str(totprijs), boeking_begin, boeking_eind, str(memberid), betaalmet]
