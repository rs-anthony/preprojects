import mysql.connector

def sendKamerinfob(type,prijs,beschrijving,foto,nummer):
    con = mysql.connector.connect(
        host="",  #port erbij indien mac
        user="",
        password="",
        database="hotel_database"
    )

    mycursor = con.cursor()

    # type = abc
    # prijs = 25
    # beschrijving = "mooie kamertje"
    # foto = "hotel.jpg"
    # nummer = 455
    sql = "INSERT INTO hotelkamer (kamertype, prijs, beschrijving, kamerfoto, kamernummer) VALUES (%s, %s, %s, %s, %s)"
    val = (type, prijs, beschrijving, foto, nummer)
    mycursor.execute(sql, val)
    #print(val,flush=True)
    con.commit()
    print(mycursor.rowcount, "kamer toegevoegd")
    return "opgeslagen"

#sendKamerinfo("test1")
