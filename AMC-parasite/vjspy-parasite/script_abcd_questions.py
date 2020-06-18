
import sqlite3

def dumps(*args):
    print(*args, sep=';')

def dump_all_answers_in_order(path):
    def do(*args):
        res = c.execute(*args).fetchall()
        if len(res)>0 and len(res[0]) == 1:
            res = [r[0] for r in res]
        return res
    c = sqlite3.connect(path+'/data/scoring.sqlite')
    c.execute('ATTACH ? AS lay', [path+'/data/layout.sqlite'])
    students = do('select DISTINCT student from layout_box')
    for s in students:
        questions = do('SELECT question FROM layout_box WHERE student = ? GROUP BY question ORDER BY page, min(ymin)', [s])
        expected = []
        for q in questions:
            ORDER = 'ABCDEFGHI'
            answers = do('''
            SELECT correct FROM layout_box b
            LEFT JOIN scoring_answer s
               ON b.student = s.student
              AND b.question = s.question
              AND b.answer = s.answer
            WHERE b.student = ?
              AND s.question = ?
            ORDER BY page, xmin, ymin
            ''', [s, q])
            expected.append(''.join([ORDER[i] for i,v in enumerate(answers) if v==1]))
        dumps(s, questions, *expected)


dump_all_answers_in_order('/home/twilight/MC-Projects/2020-network-1/')
