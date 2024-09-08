import mysql.connector
import json

def getMembergegevens(memberid):
    con = mysql.connector.connect(
        host="",  #port erbij indien mac
        user="",
        password="",
        database="hotel_database"
    )
    mycursor = con.cursor()

    mycursor.execute("SELECT * FROM member WHERE member_id="+memberid)
    myresult = mycursor.fetchall()

    jsonmemberinfo = json.dumps(myresult, indent=4, sort_keys=True, default=str)
    return jsonmemberinfo

