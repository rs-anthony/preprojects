import dbconnect

def sendMembergegevensdb(voornaam, achternaam, voorvoegsel, postcode, adres, land, tel, email, wachtwoord):
    mycursor = dbconnect.con.cursor()

    sql = "INSERT INTO member (voornaam, achternaam, voorvoegsel, adres_straatnaam_huisnummer, adres_postcode, adres_land, telefoonnummer, emailadress, wachtwoord) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = (voornaam, achternaam, voorvoegsel, adres, postcode, land, tel, email, wachtwoord)
    mycursor.execute(sql, val)
    dbconnect.con.commit()
    print("het is gelukt.")
    return "Member is opgeslagen."