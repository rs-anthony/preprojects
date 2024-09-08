class Hotelkamer:
    def __init__(self, kamertype, kamernummer, prijs, beschrijving, foto):
        self.kamertype = kamertype
        self.kamernummer = kamernummer
        self.prijs = prijs
        self.beschrijving = beschrijving
        self.foto = foto
    def __str__(self):
        return f"{self.kamertype}({self.kamernummer},{self.prijs},{self.beschrijving},{self.foto})"

k1 = Hotelkamer("1 persoonskamer", "101", 150.00, "Een mooie kamer", "https://www.dehotelpartners.nl/wp-content/uploads/2019/01/190117_Toekomst_Hotelkamer_1.jpg")
k2 = Hotelkamer("2 persoonskamer", "106", 250.00, "Nog een mooie kamer", "https://www.dehotelpartners.nl/wp-content/uploads/2019/01/190117_Toekomst_Hotelkamer_2.jpg")

print(k1)
print(k2)
