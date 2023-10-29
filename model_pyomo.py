from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import inequality

import numpy as np


def Model_Merc(nscene, nproduc, a, b, limit_demanda = 10000):
    model = ConcreteModel()

    # Seteo de indices:
    # i: productores
    # s: escenario de demanda

    # Calculando los distintos tipos de escenarios siguiendo una distribucion normal
    num_escenarios = nscene
    N = nproduc
    
    D = []
    while len(D) < 5:
        Rand = np.random.normal(0, limit_demanda, 1000)
        D = Rand[Rand >= 0]
        
    D = D[:5]

    model.nscene = N

    model.prod = Set(initialize = [i for i in range(1,N)])
    model.esc = Set(initialize = [i for i in range(1,num_escenarios)])

    # Seteo de parametros:
    # a_is , b_is coeficientes de produccion del i-esimo productor en el escenario s
    
    model.a = Param(
        model.prod,
        initialize=lambda model, i: a[i-1],
        doc='Coeficiente de producción a'
)
    
    model.b = Param(
        model.prod, 
        initialize= lambda model, i: b[i-1], 
        doc='Coeficiente de produccion b')

    
    # Seteo de variables :
    # q_is matriz de produccion donde para productor i en el escenario s

    model.q = Var(model.prod, model.esc)

    # Definiendo modelo Objetivo :
    def iso_fun(model,s):
        return sum(model.q[i,s] for i in range(1,N))

    objective_rule = iso_fun

    # Por cada escenario, una funcion objetivo
    model.Objective = Objective(
        model.esc,
        rule=objective_rule, 
        sense=maximize)
    
    # Seteando restricciones

    def mrestr1(model, p, s):
        return model.q[p,s] >= 0

    def mrestr2(model, i, j, s):
        if (model.a[i] != model.a[j] or model.b[i] != model.b[j]):
            return inequality(model.q[i, s], model.q[j, s])
        else:
            return model.q[i, s] == model.q[j, s]

    
    def mrestr3(model,p,s):
        for scene in s:
            total_dem = sum([model.q[i,scene] for i in p])
            return total_dem <= D[scene]


    model.rest1 = Constraint(model.prod, model.esc, rule = mrestr1, doc='Produccion siempre positiva')
    model.rest2 = Constraint(model.prod, model.prod, model.esc, rule = mrestr2, doc='')

    model.rest_dem = ConstraintList()
    for esc in model.esc:
        total_prod = sum(model.q[prod, esc] for prod in model.prod)
        D_stage =  D[esc]
        model.rest_dem.add(total_prod == D_stage)

    solver = SolverFactory('glpk') # Selecciona el solucionador que deseas utilizar
    result = solver.solve(model, tee=True) # Resuelve el modelo y muestra la salida

    model.display() # Muestra los valores de las variables y la función objetivo

    return result

