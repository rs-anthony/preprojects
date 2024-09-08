import mysql.connector

def sendGastgegevensdb(voornaam, achternaam, voorvoegsel, postcode, adres, land, tel, email, betaalmethode):
    con = mysql.connector.connect(
        host="",  #port erbij indien mac
        user="",
        password="",
        database="hotel_database"
    )

    mycursor = con.cursor()

    sql = "INSERT INTO member (voornaam, achternaam, voorvoegsel, adres_straatnaam_huisnummer, adres_postcode, adres_land, telefoonnummer, emailadress) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (voornaam, achternaam, voorvoegsel, adres, postcode, land, tel, email)
    mycursor.execute(sql, val)
    
    con.commit()
    sql = "SELECT * FROM member ORDER BY member_id DESC"

    mycursor.execute(sql)
    
    myresult = mycursor.fetchall()
    
    return str(myresult[0][0])