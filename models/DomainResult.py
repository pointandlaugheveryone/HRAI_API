from dataclasses import dataclass
from typing import List, Dict
from models.Suggestion import Suggestion


@dataclass
class DomainResult:
    domain: str
    occ_count: int
    occupations: List[Suggestion]


CODE_TO_DOMAIN: Dict = {
    '01': 'Generálové a důstojníci v ozbrojených silách',
    '02': 'Zaměstnanci v ozbrojených silách',
    '03': 'Zaměstnanci v ozbrojených silách, nižší hodnosti',
    '11': 'Zákonodárci, úřední ředitelé a ředitelé společností',
    '12': 'Vedoucí a řídící pracovníci ve veřejné správě a v komerční sféře',
    '13': 'Vedoucí a řídící pracovníci v průmyslu, vzdělávání a v příbuzných oborech',
    '14': 'Vedoucí a řídící pracovníci v pohostinství a službách',
    '21': 'Věda a technika',
    '22': 'Zdravotnictví',
    '23': 'Výchova a vzdělávání',
    '24': 'Obchodní sféra a veřejná správa',
    '25': 'Informační a komunikační technologie',
    '26': 'Právní, sociální a kulturní obory',
    '31': 'Věda a technika',
    '32': 'Zdravotnictví',
    '33': 'Obchodní sféra a veřejná správa',
    '34': 'Právní, sociální a kulturní obory',
    '35': 'Informační a komunikační technologie',
    '41': 'Administrativní pracovníci, sekretáři a zadávání dat',
    '42': 'Informativní služby, recepční',
    '43': 'Úředníci v logistice',
    '44': 'Úředníci',
    '51': 'Osobní služby',
    '52': 'Prodej',
    '53': 'Osobní péče v oblasti vzdělávání, zdravotnictví',
    '54': 'Pracovníci v oblasti ochrany a ostrahy',
    '61': 'Zemědělství v komerční sféře',
    '62': 'Lesnictví, rybářství a myslivost v komerční sféře',
    '63': 'Farmáři, rybáři, lovci',
    '71': 'Řemeslníci a kvalifikovaní pracovníci na stavbách',
    '72': 'Kovodělnictví, strojírnictví a příbuzné obory',
    '73': 'Uměleckých a tradiční řemesla, polygrafie',
    '74': 'Elektronika a elektrotechnika',
    '75': 'Zpracování potravin, dřeva, textilu a příbuzné obory',
    '81': 'Kvalifikovaná obsluha strojů',
    '82': 'Montážní dělníci',
    '83': 'Řidiči a doprava',
    '91': 'Uklízeči a pomocníci',
    '92': 'Pomocní pracovníci v zemědělství, lesnictví a rybářství',
    '93': 'Pomocní pracovníci v oblasti těžby, stavebnictví, výroby, dopravy',
    '94': 'Příprava jídla',
    '95': 'Pouličního prodej a služby',
    '96': 'Popeláři a jiní nekvalifikovaní pracovníci'
}
