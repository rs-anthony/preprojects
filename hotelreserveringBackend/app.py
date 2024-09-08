from flask import Flask, request
from flask_cors import CORS, cross_origin
import getKamerinfo
import sendKamerinfo
import dbreserveerkamer
import dbfetchkamer
import dbboekkamer
import sendGastgegevens
import getgeboekteKamerinfo
import checkMember
import saveMember
import memberLogin
import sendmembergegevens
import sendboekinggegevens

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/testb", methods=["POST"])
def testb():
    return "testb"

@app.route("/getKamerinfo/<zoekterm>")
def showKamerinfo(zoekterm):
    return getKamerinfo.getKamerinfo(zoekterm)

@app.route("/getgeboekteKamerinfo/<zoekterm>")
def showgeboekteKamerinfo(zoekterm):
    return getgeboekteKamerinfo.getgeboekteKamerinfo(zoekterm)

@app.route("/sendKamerinfo")
def storeKamerinfo():
    type = str(request.args.get('ftype'))
    prijs = str(request.args.get('fprijs'))
    beschrijving = str(request.args.get('fbeschrijving'))
    foto = str(request.args.get('ffoto'))
    nummer = str(request.args.get('fnummer'))
    return sendKamerinfo.sendKamerinfob(type,prijs,beschrijving,foto,nummer)

@app.route("/dbreserveerkamer/<kamerid>")
def reserveerKamer(kamerid):
    return dbreserveerkamer.sendKamerreservering(kamerid)

@app.route("/dbfetchkamer/<kamerid>")
def fetchKamer(kamerid):
    return dbfetchkamer.getSpecifickamer(kamerid)

@app.route("/sendGastgegevens", methods=["POST"])
def storeGastgegevens():
    voornaam = str(request.json['kvoornaam'])
    achternaam = str(request.json['kachternaam'])
    voorvoegsel = str(request.json['kvoorvoegsel'])
    postcode = str(request.json['kpcode'])
    adres = str(request.json['kadres'])
    land = str(request.json['kland'])
    tel = str(request.json['ktel'])
    email = str(request.json['kemail'])
    betaalmethode = str(request.json['kbetaalmethode'])
    memberid = sendGastgegevens.sendGastgegevensdb(voornaam, achternaam, voorvoegsel, postcode, adres, land, tel, email, betaalmethode)
    kamerid = str(request.json['kkamerid'])
    betaalmet = str(request.json['kbetaalmethode'])
    totprijs = str(request.json['ktotaleprijs'])
    boeking_begin = str(request.json['kstartdata'])
    boeking_eind = str(request.json['keinddata'])
    return boekKamer(kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet)
    
def boekKamer(kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet):
    return dbboekkamer.sendKamerboeking(kamerid, totprijs, boeking_begin, boeking_eind, memberid, betaalmet)

@app.route("/checkMember/<memberid>")
def dbcheckMember(memberid):
    return checkMember.checkEmailadress(memberid)

@app.route('/saveMember', methods=['POST'])
def storeMembergegevens():

    voornaam = request.json['voornaam']
    achternaam = request.json['kachternaam']
    voorvoegsel = request.json['voorvoegsel']
    postcode= request.json['postcode']
    adres = request.json['adres']
    land = request.json['land']
    tel = request.json['tel']
    email = request.json['email']
    wachtwoord = request.json['wachtwoord']
    return saveMember.sendMembergegevensdb(voornaam, achternaam, voorvoegsel, postcode, adres, land, tel, email, wachtwoord)

@app.route("/memberLogin", methods=["POST"])
@cross_origin()
def dbLoginMember():
    userEmail = str(request.json['email'])
    userPassword = str(request.json['wachtwoord'])
    return memberLogin.login(userEmail, userPassword)

@app.route("/sendmembergegevens", methods=["POST"])
def showMembergegevens():
    memberid = request.json['id']
    return sendmembergegevens.getMembergegevens(memberid)

@app.route("/sendboekinggegevens", methods=["POST"])
def showBoekinggegevens():
    memberid = request.json['id']
    return sendboekinggegevens.getBoekinggegevens(memberid)
