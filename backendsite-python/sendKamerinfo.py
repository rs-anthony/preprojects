import dbconnect

def sendKamerinfob(type,prijs,beschrijving,foto,nummer):
    mycursor = dbconnect.cursor()

    # type = abc
    # prijs = 25
    # beschrijving = "mooie kamertje"
    # foto = "hotel.jpg"
    # nummer = 455
    sql = "INSERT INTO hotelkamer (kamertype, prijs, beschrijving, kamerfoto, kamernummer) VALUES (%s, %s, %s, %s, %s)"
    val = (type, prijs, beschrijving, foto, nummer)
    mycursor.execute(sql, val)
    #print(val,flush=True)
    dbconnect.con.commit()
    print(mycursor.rowcount, "kamer toegevoegd")
    return "opgeslagen"

#sendKamerinfo("test1")
