import dbconnect
import json

def getMembergegevens(memberid):
    mycursor = dbconnect.con.cursor()

    mycursor.execute("SELECT * FROM member WHERE member_id="+memberid)
    myresult = mycursor.fetchall()

    jsonmemberinfo = json.dumps(myresult, indent=4, sort_keys=True, default=str)
    return jsonmemberinfo

