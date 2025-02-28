from random import *
from math import *
from fractions import Fraction as F

def generate_linear_equation(n):
    parts = choice(['r', 'l', 'rl'])

    if parts == 'r':
        return get_answer(f'0 = {sign_fix(gen_polynomial(1, n + 1))}')    
    elif parts == 'l':
        return get_answer(f'{sign_fix(gen_polynomial(1, n + 1))} = 0')
    else:
        nl = randint(1, n - 1)
        nr = n - nl
        
        return get_answer(f'{sign_fix(gen_polynomial(1,nl))} = {sign_fix(gen_polynomial(1,nr))}')


def gen_polynomial(mn,mx):

    n = randint(mn,mx)

    signs = [choice([' + ',' - ']) for _ in range(n)]
    signs[-1] = ''
    
    poly = ''

    for n0 in range(n):
        poly += (gen_monomial(mn_gen=1, mx_gen=10) + signs[n0])
    
    return poly

def gen_monomial(mn_gen, mx_gen):
    free = choice([True, False])

    #mono = str(choice([randrange(mn_gen, 0),randrange(1, mx_gen)]))
    mono = str(randint(mn_gen,mx_gen))
    
    if free:
        return mono
    else:
        if choice(['*',' ']) == '*':
            return mono + '*x'
        else:
            return mono + 'x'
        
def get_answer(s):

    monos = [c for c in s.replace('+ ','+').replace('- ','-').split() if c not in '+-*/^'] #=
    
    a, b = 0, 0 # ax + b = 0
    
    for m in range(0, monos.index('=')):
        if monos[m] == 'x':
            a += 1
        elif 'x' in monos[m]:
            if '*' in monos[m]:
                a += int(monos[m][:-2])
            else:
                a += int(monos[m][:-1])   
        else:
            b -= int(monos[m])
            
    for m in range(monos.index('=') + 1, len(monos)):
        if monos[m] == 'x':
            a -= 1
        elif 'x' in monos[m]:
            if '*' in monos[m]:
                a -= int(monos[m][:-2])   
            else:
                a -= int(monos[m][:-1])       
        else:
            b += int(monos[m])
            
    #print(monos, a, b)
            
    if a != 0:
        if b != 0:
            return (s, F(b, a))
        return (s, "Истина (любое число)")
    else:
        if b == 0:
            return (s, "Истина (любое число)")
        return (s, "Ложь (пустое множество)")
    
    

sign_fix = lambda s: s.replace('+ -', '- ').replace('- +', '- ').replace('- -', '+ ').replace('+ +', '+ ')

n = 4 # int(input())

eqs = {}

N = int(input("Введите кол-во генерируемых уравнений: "))

for _ in range(N):
    eq = generate_linear_equation(n)
    print(*eq, sep=' | ')
    eqs[eq[0]] = eq[1]

df = pd.DataFrame(list(eqs.items()), columns=["Equations","Answers"])
df.to_csv("equations (gen).csv", index=False, encoding="utf-8")
print("✅ Уравнения сохранены в equations (gen).csv!")
