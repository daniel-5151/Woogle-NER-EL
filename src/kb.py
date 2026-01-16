import sqlite3

def index_kb(kb):
    db = sqlite3.connect(':memory:')
    cur = db.cursor()

    # Create surface form table
    cur.execute('create virtual table kb using fts5(qid UNINDEXED, surface_form, normal_form UNINDEXED, label UNINDEXED, description UNINDEXED, tokenize="porter unicode61");')

    # populate form table table
    cur.executemany(
        'insert into kb (qid, surface_form, normal_form, label, description) values (?,?,?,?,?);',
        kb[['qid', 'surface_form', 'normal_form','label', 'description']].to_records(index=False))
    db.commit()
    return db, cur