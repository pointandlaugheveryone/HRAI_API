### Endpointy

| method | path              |                                                                                                                                                   |
|--------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| POST   | `/resume/skills`  | PDF <br/>-> návrhy dalších dovedností                                                                                                             |
| POST   | `/resume/domains` | PDF <br/>-> informace o možnostech na pozicích v oblastech pracovního trhu                                                                        |
| POST   | `/text/skills`    | text <br/>-> návrhy dalších dovedností                                                                                                            |
| POST   | `/text/domains`   | text (dovednosti + pracovní zkušenosti) <br/>-> informace o možnostech na pozicích v oblastech pracovního trhu                                    |
| POST   | `/query`          | text + jestli hledat 'skill', 'occupation', 'isco_group' nebo 'skill_group', <br/>případně minimální podobnost pro rozpoznání <br/>-> výsledek dle databáze |

další parametry lze nastavit v `config.py`