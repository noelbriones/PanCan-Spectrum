import pymysql as sql


def list_chem_compounds(mz_list):

    # Open database connection
    db = sql.connect("localhost", "root", "", "chemcompoundsdb")

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    final_list = []
    for mz in mz_list:
        cursor.execute("""SELECT `names`.name FROM `names` 
                         WHERE ((`names`.compound_id IN (SELECT `chemical_data`.compound_id FROM `chemical_data` WHERE `chemical_data`.type="mass")) 
                            AND (`names`.compound_id IN (SELECT `chemical_data`.compound_id FROM `chemical_data` WHERE `chemical_data`.chemical_data LIKE '{}%')))
                                AND (`names`.name in (SELECT `compounds`.name FROM `compounds`));""".format(mz))
        temp_list = cursor.fetchall()
        for x in temp_list:
            final_list.append(x)
    print(final_list)

    # disconnect from server
    db.close()

    return final_list
