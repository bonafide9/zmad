class SystemEkspertowyOceny:
    def __init__(self, jasnosc, przygotowanie, punktualnosc, kontakt, sposob, dostepnosc, atrakcyjnosc):
        self.jasnosc = jasnosc
        self.przygotowanie = przygotowanie
        self.punktualnosc = punktualnosc
        self.kontakt = kontakt
        self.sposob = sposob
        self.dostepnosc = dostepnosc
        self.atrakcyjnosc = atrakcyjnosc

    def ewaluuj(self):
        oceny = []
        zalecenia = []

        
        if self.jasnosc >= 4 and self.przygotowanie >= 4 and self.kontakt >= 4:
            oceny.append("bardzo dobra")
            
        if self.jasnosc <= 2 and self.kontakt <= 2:
            oceny.append("słaba")
      
        if self.punktualnosc >= 4 and self.przygotowanie >= 4 and self.sposob >= 4:
            oceny.append("dobra")

        if self.jasnosc == 3 and self.kontakt == 3 and self.atrakcyjnosc == 3:
            oceny.append("przeciętna")

        if self.sposob <= 2 or self.dostepnosc <= 2:
            zalecenia.append("poprawić komunikację i zasady oceniania")

        if self.punktualnosc <= 2:
            zalecenia.append("zwrócić uwagę na punktualność")
            
        if self.atrakcyjnosc >= 4 and self.jasnosc >= 4:
            zalecenia.append("pochwała za angażujące zajęcia")
            
        if self.kontakt == 1 and self.dostepnosc == 1:
            oceny.append("krytycznie słaba")
            
        if self.przygotowanie >= 3 and self.punktualnosc >= 3 and self.sposob >= 3:
            zalecenia.append("solidne podstawy organizacyjne")
            
        if self.atrakcyjnosc <= 2 and self.jasnosc <= 2:
            zalecenia.append("konieczna zmiana metod dydaktycznych")

        ocena_koncowa = oceny[-1] if oceny else "brak wyraźnej oceny (wymaga dopytania)"
        
        return {
            "Ocena prowadzącego": ocena_koncowa,
            "Zalecenia": zalecenia if zalecenia else ["Brak uwag"]
        }

system = SystemEkspertowyOceny(
    jasnosc=5,
    przygotowanie=4,
    punktualnosc=5,
    kontakt=4,
    sposob=4,
    dostepnosc=3, 
    atrakcyjnosc=5
)

wynik = system.ewaluuj()

print("=== WYNIKI EWALUACJI ===")
print(f"Ocena: {wynik['Ocena prowadzącego']}")
print("Zalecenia:")
for z in wynik['Zalecenia']:
    print(f"- {z}")