import dbconnect

def sendGastgegevensdb(voornaam, achternaam, voorvoegsel, postcode, adres, land, tel, email, betaalmethode):
    mycursor = dbconnect.con.cursor()

    sql = "INSERT INTO member (voornaam, achternaam, voorvoegsel, adres_straatnaam_huisnummer, adres_postcode, adres_land, telefoonnummer, emailadress) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (voornaam, achternaam, voorvoegsel, adres, postcode, land, tel, email)
    mycursor.execute(sql, val)
    
    dbconnect.con.commit()
    sql = "SELECT * FROM member ORDER BY member_id DESC"

    mycursor.execute(sql)
    
    myresult = mycursor.fetchall()
    
    return str(myresult[0][0])