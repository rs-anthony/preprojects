import hashlib
import json
import mysql.connector

def signup(userEmail, userPassword, confirmUserpassword):
    foutPassword = False
    if confirmUserpassword == userPassword:
        enc = userPassword.encode()
        hash1 = hashlib.md5(enc).hexdigest()     

        #Send userinfo to database

    else:
        foutPassword = True
        print("Wachtwoord matcht niet!")
    return 