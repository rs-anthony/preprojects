import dbconnect

def sendKamerboeking(kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet):
    mycursor = dbconnect.con.cursor()
    
    sql = "INSERT INTO boeking (kamer_id, totaalprijs, boekingsdatum_begin, boekingsdatum_eind, member_id, betaalmethode) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet)
    print(val)
    mycursor.execute(sql, val)

    dbconnect.con.commit()

    sql = "SELECT * FROM boeking ORDER BY boeking_id DESC"   

    mycursor.execute(sql)
    
    myresult = mycursor.fetchall()
    return [str(myresult[0][0]), str(kamerid), str(totprijs), boeking_begin, boeking_eind, str(memberid), betaalmet]
